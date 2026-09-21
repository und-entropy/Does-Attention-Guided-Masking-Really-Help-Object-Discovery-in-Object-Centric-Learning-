"""Re-evaluate the paper's six-dataset, two-strategy, three-seed comparison.

Input: datasets, RM/AGM checkpoints, and the corresponding training logs; CUDA is required.
Output: per-model evaluation records and an aggregate file under the configured output directory.
Scope: the clean primary comparison only; no additional ablations are run.
"""
from pathlib import Path
import argparse
import gc
import pandas as pd
import torch as pt
from common_AGM import (DATASET_BY_KEY, add_common_args, project_paths, cfg_for, data_for,
                        weight_for, choose_checkpoint, seed_evaluation, build_eval_components,
                        build_model, move_batch_to_cuda, Config, METRICS)


@pt.inference_mode()
def evaluate(cfg_path, ckpt, data_dir, workers, sample_limit):
    seed_evaluation(0)
    cfg, dataset, loader, loss_fn, acc_fn, callbacks = build_eval_components(
        cfg_path, data_dir, batch_size=1, num_workers=workers, sample_limit=sample_limit)
    model = build_model(cfg, ckpt)
    seed_evaluation(0)
    pack = Config({'dataset_v': loader, 'model': model, 'loss_fn_v': loss_fn, 'acc_fn_v': acc_fn,
                   'callback_v': callbacks, 'epoch': 0, 'isval': True})
    sums, counts = {m: 0.0 for m in METRICS}, {m: 0 for m in METRICS}
    for cb in callbacks:
        cb.before_epoch(**pack)
    for batch in loader:
        pack.batch = move_batch_to_cuda(batch)
        for cb in callbacks:
            cb.before_step(**pack)
        with pt.autocast('cuda', enabled=True):
            pack.output = model(**pack)
            for cb in callbacks:
                cb.after_forward(**pack)
            pack.loss = loss_fn(**pack)
        pack.acc = acc_fn(**pack)
        for m in METRICS:
            # MetricWrap returns (metric, valid); boolean validity is never averaged as a score.
            metric, valid = pack.acc[m]
            values = metric[valid].detach().double()
            sums[m] += float(values.sum().item())
            counts[m] += int(values.numel())
        for cb in callbacks:
            cb.after_step(**pack)
    for cb in callbacks:
        cb.after_epoch(**pack)
    if not all(counts.values()):
        raise ValueError('某指标没有有效样本')
    out = {m: sums[m] / counts[m] for m in METRICS}
    out['n_eval_samples'] = len(dataset)
    del model, pack, dataset, loader
    gc.collect()
    pt.cuda.empty_cache()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument('--sample_limit', type=int, default=256)
    args = parser.parse_args()
    if args.max_batches is not None:
        raise ValueError('正式主对比不使用 max_batches，请用独立 sample_limit 调试')
    paths = project_paths(args)
    out = paths['out_root'] / 'primary_comparison'
    out.mkdir(parents=True, exist_ok=True)
    for key in args.datasets:
        spec = DATASET_BY_KEY[key]
        for strategy in ['rand', 'pgm']:
            for seed in args.seeds:
                selected = choose_checkpoint(weight_for(paths, spec, strategy), seed)
                if selected['reason'] != 'best_available_log_epoch':
                    raise RuntimeError(f'{key}/{strategy}/{seed}: 需要可与训练日志精确匹配的 checkpoint，当前 {selected["reason"]}')
                scores = evaluate(cfg_for(paths, spec), selected['ckpt'], data_for(paths, spec),
                                  args.num_workers if args.num_workers is not None else 0, args.sample_limit)
                row = {'dataset': key, 'dataset_label': spec.label, 'strategy': strategy, 'seed': seed,
                       'checkpoint_epoch': selected['best_epoch'], 'checkpoint': str(selected['ckpt']),
                       'protocol_version': 'agm-clean-balanced256-valid-metrics', **scores}
                pd.DataFrame([row]).to_csv(out / f'{key}_{strategy}_{seed}_AGM.csv', index=False)
                print(key, strategy, seed, scores, flush=True)
    files = [out / f'{d}_{p}_{s}_AGM.csv' for d in args.datasets for p in ['rand','pgm'] for s in args.seeds]
    pd.concat([pd.read_csv(p) for p in files], ignore_index=True).to_csv(out / 'primary_AGM.csv', index=False)


if __name__ == '__main__':
    main()
