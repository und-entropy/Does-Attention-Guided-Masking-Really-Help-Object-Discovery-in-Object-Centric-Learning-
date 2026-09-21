"""Generate the paper's main and absolute-result tables from real evaluations.

Input: six datasets, two strategies, and three seeds in ``data/primary_AGM.csv``.
Output: summary metrics, paired differences, and LaTeX tables. Use
``--check_paper`` to compare the final paper tables with the clean log excerpt.
"""
from pathlib import Path
import argparse
import ast
import json
import re
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATASETS = ['clevrtex', 'coco', 'voc', 'ytvis', 'movi_c', 'movi_e']
LABELS = dict(zip(DATASETS, ['CLEVRTex', 'COCO', 'VOC', 'YTVIS-2022', 'MOVi-C', 'MOVi-E']))
METRICS = ['ari', 'ari_fg', 'mbo', 'miou']
HEADERS = ['ARI', 'FG-ARI', 'mBO', 'mIoU']


def validate(frame, keys, expected=None):
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError('输入为空或存在重复实验记录')
    if expected is not None and len(frame) != expected:
        raise ValueError(f'需要 {expected} 条记录，实际 {len(frame)} 条')
    if not np.isfinite(frame[METRICS].to_numpy(float)).all():
        raise ValueError('存在非有限指标')


def original_tables():
    delta = (HERE / 'paper table/tab_delta_AGM.tex').read_text(encoding='utf-8')
    absolute = (HERE / 'paper table/tab_objdiscov_absolute_AGM.tex').read_text(encoding='utf-8')
    expected = {}
    for dataset, label in LABELS.items():
        match = re.search(re.escape(label) + r'\s*&(?P<cells>.*?)(?=\\\\)', delta, re.S)
        cells = re.findall(r'\$([+-]?\d+\.\d+)_\{\\pm(\d+\.\d+)\}\$', match['cells'])
        for metric, (mean, sd) in zip(METRICS, cells):
            expected[(dataset, 'delta', metric)] = (float(mean), float(sd))
    current = None
    reverse = {v: k for k, v in LABELS.items()}
    for line in absolute.splitlines():
        if not re.search(r'&\s*(RM|AGM)\s*&', line):
            continue
        parts = line.split('&')
        current = parts[0].strip() or current
        method = 'rand' if parts[1].strip() == 'RM' else 'pgm'
        cells = re.findall(r'\$([+-]?\d+\.\d+)\\pm(\d+\.\d+)\$', line)
        for metric, (mean, sd) in zip(METRICS, cells):
            expected[(reverse[current], method, metric)] = (float(mean), float(sd))
    if len(expected) != 72:
        raise ValueError('论文参考表解析不完整')
    return expected


def verify_clean_log(frame):
    text = (HERE / 'data/clean_evaluation_AGM.txt').read_text(encoding='utf-8')
    records, context = {}, None
    for line in text.splitlines():
        match = re.search(r'\[start-input \d+/\d+ (rand|pgm)/([^/]+)/s(\d+)/sigma=0(?:\.0)?\]', line)
        if match:
            context = (match[2], match[1], int(match[3]))
        if line.startswith('0-val '):
            if context in records or context is None:
                raise ValueError('评估日志的上下文重复或缺失')
            records[context] = ast.literal_eval(line[6:])
    if len(records) != 36:
        raise ValueError('需要 36 条原始 clean-evaluation 记录')
    for row in frame.itertuples():
        values = records[(row.dataset, row.strategy, int(row.seed))]
        for metric in METRICS:
            if abs(float(getattr(row, metric)) - float(values[metric])) > 1e-12:
                raise ValueError(f'原始日志与 CSV 不一致：{row.dataset}/{row.strategy}/{row.seed}/{metric}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / 'data/primary_AGM.csv')
    parser.add_argument('--out_dir', type=Path, default=HERE / 'outputs/tables')
    parser.add_argument('--check_paper', action='store_true', help='核对最终论文表格及所附原始日志摘录')
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    validate(frame, ['dataset', 'strategy', 'seed'], 36)
    design = {(d, p, s) for d in DATASETS for p in ['rand', 'pgm'] for s in [42, 43, 44]}
    if set(frame[['dataset', 'strategy', 'seed']].itertuples(index=False, name=None)) != design:
        raise ValueError('六域三种子实验组合不完整')
    wide = frame.pivot(index=['dataset', 'seed'], columns='strategy', values=METRICS)
    paired = pd.DataFrame({m: 100 * (wide[m]['pgm'] - wide[m]['rand']) for m in METRICS}).reset_index()
    summary = []
    absolute_rows, delta_rows = [], []
    reference = original_tables() if args.check_paper else {}
    largest_difference = 0.0
    for dataset in DATASETS:
        for method in ['rand', 'pgm', 'delta']:
            group = paired[paired.dataset == dataset] if method == 'delta' else frame[(frame.dataset == dataset) & (frame.strategy == method)]
            factor = 1 if method == 'delta' else 100
            values = {m: (float(group[m].mean() * factor), float(group[m].std(ddof=1) * factor)) for m in METRICS}
            if args.check_paper:
                for m, (mean, sd) in values.items():
                    a, b = reference[(dataset, method, m)]
                    error = max(abs(mean - a), abs(sd - b))
                    largest_difference = max(largest_difference, error)
                    if error > 0.00501:
                        raise ValueError(f'与论文两位小数表格不一致：{dataset}/{method}/{m}: {values[m]} vs {(a,b)}')
            for metric, (mean, sd) in values.items():
                summary.append({'dataset': dataset, 'method': method, 'metric': metric, 'mean': mean, 'sample_std': sd})
            cells = [f'${a:+.2f} \\pm {b:.2f}$' if method == 'delta' else f'${a:.2f} \\pm {b:.2f}$' for a, b in values.values()]
            row = LABELS[dataset] + ' & ' + (' & '.join(cells) if method == 'delta' else ('RM' if method == 'rand' else 'AGM') + ' & ' + ' & '.join(cells)) + r' \\'
            (delta_rows if method == 'delta' else absolute_rows).append(row)
    if args.check_paper:
        verify_clean_log(frame)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(args.out_dir / 'summary_AGM.csv', index=False)
    paired.to_csv(args.out_dir / 'paired_differences_AGM.csv', index=False)
    for name, columns, header, rows in [
        ('tab_delta_AGM.tex', 'lrrrr', 'Dataset & ' + ' & '.join(HEADERS), delta_rows),
        ('tab_absolute_AGM.tex', 'llrrrr', 'Dataset & Method & ' + ' & '.join(HEADERS), absolute_rows),
    ]:
        lines = [r'\begin{tabular}{' + columns + '}', r'\toprule', header + r' \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}']
        (args.out_dir / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    report = {'models': len(frame), 'paired_comparisons': len(paired),
              'fg_ari_decreased_pairs': int((paired.ari_fg < 0).sum()),
              'paper_and_log_checked': args.check_paper, 'largest_rounding_difference_pp': largest_difference}
    (args.out_dir / 'check_AGM.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
