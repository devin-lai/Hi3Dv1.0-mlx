"""Plot the six recorded FP32/FP16 observations; does not execute inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'assets/benchmarks.png')
    args = parser.parse_args()
    data = json.loads((ROOT / 'benchmarks/benchmark-512.json').read_text())
    runs = {row['run']: row for row in data['runs']}
    cases = ['dragon', 'bust', 'locomotive']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor='#f0f3fa')
    colors = ['#8090a5', '#009e86']
    for ax, key, title, unit in zip(
            axes, ['total_s', 'peak_gb'],
            ['Refinement latency', 'Parent-process peak memory'], ['Seconds ↓', 'GiB ↓']):
        ax.set_facecolor('#f0f3fa')
        y = np.arange(len(cases))
        for offset, config, label, color in zip(
                [-0.19, 0.19], ['torch-fp32', 'mlx-fp16'],
                ['PyTorch-MPS FP32', 'MLX FP16'], colors):
            values = [runs[f'{case}-512-{config}'][key] for case in cases]
            bars = ax.barh(y + offset, values, height=0.34, label=label, color=color)
            ax.bar_label(bars, fmt='%.1f', padding=4, fontsize=9)
        ax.set_yticks(y, [c.capitalize() for c in cases])
        ax.invert_yaxis()
        ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
        ax.set_xlabel(unit)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.grid(axis='x', alpha=0.12)
        ax.set_axisbelow(True)
        ax.margins(x=0.16)
    axes[0].legend(frameon=False, fontsize=9, loc='lower right')
    fig.suptitle('512³ refinement on Apple M5 · 32 GiB', fontsize=17, fontweight='bold')
    fig.text(0.5, 0.025,
             'One historical run per case/configuration · identical saved coarse meshes · 25 Heun steps\n'
             'Includes loading and conditioning; excludes coarse generation, downloads and final export. '
             'Memory excludes child encoders.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, 0.10, 1, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
