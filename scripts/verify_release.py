"""Check the public source inventory, release boundary and local documentation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', '.venv', '.cache', '__pycache__', '.pytest_cache', 'build', 'dist'}
CHECKPOINT_SUFFIXES = {'.safetensors', '.ckpt', '.pt', '.pth', '.bin'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--update-manifest', action='store_true',
                        help='Record intentionally reviewed source/test changes.')
    args = parser.parse_args()
    paths = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and not any(part in IGNORED or part.endswith('.egg-info')
                               for part in p.relative_to(ROOT).parts))
    errors = []
    source = {}
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in CHECKPOINT_SUFFIXES:
            errors.append(f'Checkpoint in source release: {rel}')
        if path.name.startswith('.env') or path.suffix in {'.pem', '.key'}:
            errors.append(f'Credential/config file in source release: {rel}')
        if any('convert' in part.lower() for part in path.relative_to(ROOT).parts):
            errors.append(f'Conversion tooling path in source release: {rel}')
        if path.stat().st_size > 25 * 1024**2:
            errors.append(f'Unexpected large source-repository file: {rel}')
        if path.suffix == '.py':
            try:
                tree = ast.parse(path.read_text(), filename=rel)
                if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and n.name in {'quantize_weight', 'convert_checkpoint'}
                       for n in ast.walk(tree)):
                    errors.append(f'Weight-conversion function in source release: {rel}')
            except SyntaxError as exc:
                errors.append(f'{rel}: {exc.msg}')
            if path.relative_to(ROOT).parts[0] in {'src', 'tests'}:
                source[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix == '.md':
            text = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
            targets = re.findall(r'!?\[[^\]]*\]\(([^\s)]+)', text)
            targets += re.findall(r'(?:href|src)="([^"]+)"', text)
            for target in targets:
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                destination = (path.parent / unquote(parsed.path)).resolve()
                if not destination.is_relative_to(ROOT) or not destination.exists():
                    errors.append(f'Broken local link in {rel}: {target}')
    manifest_path = ROOT / 'runtime-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if args.update_manifest and not errors:
        manifest['files'] = source
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if source != manifest['files']:
        changed = sorted(k for k in set(source) | set(manifest['files'])
                         if source.get(k) != manifest['files'].get(k))
        errors.append('Source inventory changed; review before updating manifest: '
                      + ', '.join(changed))
    records = json.loads((ROOT / 'benchmarks/benchmark-512.json').read_text())
    run_ids = {row['run'] for row in records['runs']}
    provenance = json.loads((ROOT / 'assets/provenance.json').read_text())
    if not set(provenance['chart_runs']) <= run_ids:
        errors.append('A chart run is missing from the benchmark evidence.')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'Checked {len(source)} inference/test files, release contents, '
          f'local documentation links and {len(run_ids)} benchmark records.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
