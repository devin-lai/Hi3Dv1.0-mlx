"""Numerical and lifetime contracts of the additional FP16 optimizations."""
import os
import sys
from pathlib import Path
os.environ.setdefault("MLX_ENABLE_TF32", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pytest
mx = pytest.importorskip("mlx.core")
from hi3d_mac.mlx_backend import kernels
from hi3d_mac.mlx_backend.transformer import Precision, Sparc3DTransformerMLX, Weights


@pytest.mark.parametrize("dtype", [mx.float16, mx.float32])
@pytest.mark.parametrize("group", [8, 16, 32])
def test_simd_qkv_float64_oracle(dtype, group):
    # Odd sequence length and head count exercise partial threadgroups.
    rng = np.random.default_rng(2026)
    b,n,h,d = 2,37,3,96
    raw = rng.normal(0,2,(b*n,3*h*d)).astype(np.float16 if dtype == mx.float16 else np.float32)
    phases = rng.uniform(-3,3,(n,d//2)).astype(np.float32)
    c,s = np.cos(phases),np.sin(phases)
    gq,gk = (rng.uniform(.5,1.5,(h,d)).astype(np.float32) for _ in range(2))
    out = kernels.qkv_rope_rms_simd(*map(mx.array,(raw,c,s,gq,gk)),b,h,d,group)
    x = raw.astype(np.float64).reshape(b,n,3,h,d//2,2)
    for part,g in ((0,gq),(1,gk)):
        z = (x[:,:,part,...,0] + 1j*x[:,:,part,...,1]) * (c+1j*s)[None,:,None,:]
        y = np.stack((z.real,z.imag),-1).reshape(b,n,h,d)
        y = y / np.linalg.norm(y,axis=-1,keepdims=True) * g * np.sqrt(d)
        got = np.array(out[part]).transpose(0,2,1,3).astype(np.float64)
        assert np.linalg.norm(got-y)/np.linalg.norm(y) < (2.2e-4 if dtype == mx.float16 else 3e-7)
    assert np.array_equal(np.array(out[2]).transpose(0,2,1,3),raw.reshape(b,n,3,h,d)[:,:,2])


def test_conditioning_release_restore_exact(tmp_path):
    rng=np.random.default_rng(2)
    p={}
    for name,ni,no,dtype in (("t_embedder.mlp.0",256,4,np.float32),
                             ("t_embedder.mlp.2",4,4,np.float32),
                             ("adaLN_modulation.1",4,24,np.float32),
                             ("blocks.0.cross_attn.to_kv",3,8,np.float16),
                             ("blocks.0.cross_attn.to_out",4,4,np.float16)):
        p[name+'.weight']=mx.array(rng.normal(0,.1,(no,ni)).astype(dtype))
        p[name+'.bias']=mx.array(rng.normal(0,.1,(no,)).astype(dtype))
    path=tmp_path/'model.safetensors'
    mx.save_safetensors(str(path),p)
    model=Sparc3DTransformerMLX.__new__(Sparc3DTransformerMLX)
    model.w=Weights(p,Precision());model.dt=mx.float16
    model.weights_path=path;model._released_names=set();model._schedule_cache={}
    model._prepared_condition=model._prepared_kv=model._negative_cross=None
    model.num_blocks=model.heads=1;model.head_dim=model.channels=4
    cond=rng.normal(size=(1,7,3)).astype(np.float32)
    refs={t:tuple(np.array(a) for a in model.embed(t)) for t in (1.04,.83,.5)}
    expected_kv=[tuple(np.array(a) for a in pair) for pair in model.context_kv(mx.array(cond))]
    expected_neg=[np.array(a) for a in model.negative_cross()]
    kv=model.prepare_pass(cond,[1.04,.83,1.04])
    assert model.preparation_stats['cached_times']==2
    assert not any(k.startswith('t_embedder.') or '.cross_attn.to_kv.' in k for k in model.w.p)
    for t in (1.04,.83):
        for a,b in zip(model.embed(t),refs[t]):assert np.array_equal(np.array(a),b)
    for a,b in zip(kv[0],expected_kv[0]):assert np.array_equal(np.array(a),b)
    assert np.array_equal(np.array(model.negative_cross()[0]),expected_neg[0])
    # A new, uncached time restores the released source tensors, never rounds
    # the requested sigma to its nearest cached neighbour.
    for a,b in zip(model.embed(.5),refs[.5]):assert np.array_equal(np.array(a),b)
    kv2=model.prepare_pass(cond,[.5])
    for a,b in zip(kv2[0],expected_kv[0]):assert np.array_equal(np.array(a),b)
    changed=cond*1.5
    newkv=model.prepare_pass(changed,[.83])
    direct=mx.addmm(p['blocks.0.cross_attn.to_kv.bias'],mx.array(changed[0]).astype(mx.float16),
                   p['blocks.0.cross_attn.to_kv.weight'].T).reshape(7,2,1,4)
    assert np.array_equal(np.array(newkv[0][0]),np.array(direct[:,0].transpose(1,0,2)[None]))


def test_shared_first_attention_preserves_branches():
    rng=np.random.default_rng(3);c,h,d,n=192,2,96,77
    model=Sparc3DTransformerMLX.__new__(Sparc3DTransformerMLX)
    model.precision=Precision();model.dt=mx.float16;model.heads=h;model.head_dim=d
    model.compile=True;model._block_qmeta={};model._block_fns={}
    w={}
    for name,ni,no in (("self_attn.to_qkv",c,3*c),("self_attn.to_out",c,c),
                       ("cross_attn.to_q",c,c),("cross_attn.to_out",c,c),
                       ("mlp.mlp.0",c,4*c),("mlp.mlp.2",4*c,c)):
        w[name+'.weight']=mx.array(rng.normal(0,.025,(no,ni)).astype(np.float16))
        w[name+'.bias']=mx.array(rng.normal(0,.01,(no,)).astype(np.float16))
    w.update({'modulation':mx.zeros((1,6*c)), 'norm2.weight':mx.ones((c,)), 'norm2.bias':mx.zeros((c,)),
              'self_attn.q_rms_norm.gamma':mx.ones((h,d)), 'self_attn.k_rms_norm.gamma':mx.ones((h,d))})
    model.block_weights=[w]
    x=mx.array(rng.normal(size=(1,n,c)).astype(np.float16));x=mx.broadcast_to(x,(2,n,c))
    mod=mx.array(rng.normal(0,.1,(1,6*c)).astype(np.float16))
    phases=mx.array(rng.uniform(-3,3,(n,d//2)).astype(np.float32))
    kv=tuple(mx.array(rng.normal(size=(1,h,19,d)).astype(np.float16)) for _ in range(2))
    neg=mx.array(rng.normal(0,.01,(1,c)).astype(np.float16))
    args=(0,x,mod,mx.cos(phases),mx.sin(phases),kv,neg)
    ref=model.run_block(*args);got=model.run_block(*args,shared_input=True)
    assert np.array_equal(np.array(ref),np.array(got))
    attention_only=model.run_block(*args,shared_attention=True)
    assert np.array_equal(np.array(ref),np.array(attention_only))


def test_residual_cache_guards_and_pass_reset(monkeypatch):
    from hi3d_mac.mlx_backend.residual_cache import CachedTransformer

    called=[]
    def block(self,i,x,mod,*args,**kwargs):
        called.append(i)
        return x if i==0 else (x+mx.array(i,dtype=x.dtype))
    def forward(self,feats,t,geo,kvs):
        x=feats
        for i in range(self.num_blocks):x=self.run_block(i,x,self.test_mod)
        return [x]
    monkeypatch.setattr(Sparc3DTransformerMLX,'run_block',block)
    monkeypatch.setattr(Sparc3DTransformerMLX,'__call__',forward)
    model=CachedTransformer.__new__(CachedTransformer)
    model.num_blocks=3;model.dt=mx.float16;model.threshold=.012
    model.test_mod=mx.ones((1,6),mx.float16);model.reset_cache()
    x=mx.ones((1,7,4),mx.float16)
    def run(value,t=.5):
        called.clear();out=model(value,t,None,[[]])[0];mx.eval(out)
        return list(called)
    assert run(x)==[0,1,2]
    assert run(x*mx.array(1.001,mx.float16))==[0]
    assert run(x)==[0,1,2]  # never two consecutive cached evaluations
    assert run(x*2)==[0,1,2]  # first-block change fails the threshold
    model.test_mod*=2
    assert run(x*2)==[0,1,2]  # modulation changes independently
    assert run(mx.ones((1,9,4),mx.float16))==[0,1,2]  # support/shape changed
    assert run(x,t=.8)==[0,1,2]  # high sigma is never reused
    assert model._anchor is None
    assert run(x)==[0,1,2]
    model.reset_cache()
    assert model.cache_calls==0 and model.cache_events==[] and model._residual is None
    assert run(x)==[0,1,2]
