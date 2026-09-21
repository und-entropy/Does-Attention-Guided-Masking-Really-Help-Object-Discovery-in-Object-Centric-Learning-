"""
Copyright (c) 2024 Genera1Z
https://github.com/Genera1Z
"""
from argparse import ArgumentParser
from pathlib import Path
import gc
import json
import os
import random
import shutil
import time

import numpy as np
import torch as pt
import tqdm

from object_centric_bench.datum import DataLoader
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap
from object_centric_bench.util import Config, build_from_config

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # reproducibility
pt._dynamo.config.suppress_errors = True  # one_hot, interplolate


def memory_report(tag):
    values = {}
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value = line.split(":", 1)
                values[key] = value.strip()
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                values["MemAvailable"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if pt.cuda.is_available():
        values["cuda_allocated_mib"] = round(pt.cuda.memory_allocated() / 2**20, 1)
        values["cuda_reserved_mib"] = round(pt.cuda.memory_reserved() / 2**20, 1)
    print(f"[memory:{tag}] {values}", flush=True)


def train_epoch(pack):
    t0 = time.time()
    pack.model.train()
    pack.isval = False
    [_.before_epoch(**pack) for _ in pack.callback_t]

    for batch in tqdm.tqdm(pack.dataset_t):
        if pack.step_count + 1 > pack.total_step:
            break
        pack.batch = batch

        [_.before_step(**pack) for _ in pack.callback_t]
        with pt.autocast("cuda", enabled=True):
            pack.output = pack.model(**pack)
            [_.after_forward(**pack) for _ in pack.callback_t]
            pack.loss = pack.loss_fn_t(**pack)  # {k:(loss,valid),..}
        # for pack.loss/acc
        # - value: dtype=float, shape=(b,). but actually (b=1,) for loss
        # - valid: dtype=bool, shape=(b,). but actually (b=1,) for loss
        pack.acc = pack.acc_fn_t(**pack)  # in autocast may cause inf

        flag = True
        for loss_i, valid_i in pack.loss.values():
            if valid_i.sum() == 0:
                print("no valid sample in batch")  # then will not back prop
                flag = False
                break

        if flag:
            with pt.autocast("cuda", enabled=True):
                loss_mean_sum = sum(_l[_v].mean() for _l, _v in pack.loss.values())

            pack.optimiz.zero_grad()
            pack.optimiz.gscale.scale(loss_mean_sum).backward()
            if pack.optimiz.gclip is not None:
                pack.optimiz.gscale.unscale_(pack.optimiz)
                pack.optimiz.gclip(pack.model.parameters())
            pack.optimiz.gscale.step(pack.optimiz)
            pack.optimiz.gscale.update()

        [_.after_step(**pack) for _ in pack.callback_t]

        pack.step_count += 1

    [_.after_epoch(**pack) for _ in pack.callback_t]
    print("b/s:", len(pack.dataset_t) / (time.time() - t0))


@pt.inference_mode()
def val_epoch(pack):
    pack.model.eval()
    pack.isval = True
    [_.before_epoch(**pack) for _ in pack.callback_v]

    for batch in pack.dataset_v:
        pack.batch = batch

        [_.before_step(**pack) for _ in pack.callback_v]

        with pt.autocast("cuda", enabled=True):
            pack.output = pack.model(**pack)
            [_.after_forward(**pack) for _ in pack.callback_v]
            pack.loss = pack.loss_fn_v(**pack)
        pack.acc = pack.acc_fn_v(**pack)  # in autocast may cause inf

        [_.after_step(**pack) for _ in pack.callback_v]

    [_.after_epoch(**pack) for _ in pack.callback_v]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    pt.manual_seed(seed)


def run_single_seed(
    seed, cfg_file, data_path, save_dir, ckpt_file,
    ranking_mode="guided", ranking_alpha=None, num_workers=None, low_memory=False,
):
    cfg_name = cfg_file.stem.removesuffix("_AGM")
    cfg = Config.fromfile(cfg_file)
    if num_workers is not None:
        cfg.num_work = int(num_workers)
    if hasattr(cfg.model, "decode"):
        cfg.model.decode.ranking_mode = ranking_mode
        if ranking_alpha is not None:
            cfg.model.decode.alpha = float(ranking_alpha)

    # （默认）：分层结构 save/cfg_name/seed42/ （推荐，方便统一管理同配置多种子）
    save_path = Path(save_dir) / cfg_name / f"seed{seed}"

    save_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(cfg_file, save_path.parent)
    set_seed(seed)
    pt.backends.cudnn.benchmark = False
    pt.backends.cudnn.deterministic = True
    pt.use_deterministic_algorithms(True, warn_only=True)

    work_init_fn = lambda _: set_seed(seed)
    rng = pt.Generator()
    rng.manual_seed(seed)

    cfg.dataset_t.base_dir = cfg.dataset_v.base_dir = data_path

    dataset_t = build_from_config(cfg.dataset_t)
    pin_memory = cfg.num_work > 0 and not low_memory
    loader_memory_kwargs = {}
    if low_memory and cfg.num_work > 0:
        loader_memory_kwargs["prefetch_factor"] = 1
        loader_memory_kwargs["persistent_workers"] = False
    print(
        f"[training resources] num_workers={cfg.num_work}, "
        f"pin_memory={pin_memory}, prefetch_factor={loader_memory_kwargs.get('prefetch_factor', 'default')}, "
        f"batch_t={cfg.batch_size_t}, batch_v={cfg.batch_size_v}",
        flush=True,
    )
    dataload_t = DataLoader(
        dataset_t,
        cfg.batch_size_t,
        shuffle=True,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_t),
        pin_memory=pin_memory,
        worker_init_fn=work_init_fn,
        generator=rng,
        **loader_memory_kwargs,
    )
    dataset_v = build_from_config(cfg.dataset_v)
    dataload_v = DataLoader(
        dataset_v,
        cfg.batch_size_v,
        shuffle=False,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=pin_memory,
        worker_init_fn=work_init_fn,
        generator=rng,
        **loader_memory_kwargs,
    )
    model = build_from_config(cfg.model)
    print(model)
    model = ModelWrap(model, cfg.model_imap, cfg.model_omap)

    if ckpt_file:
        if isinstance(ckpt_file, (list, tuple)):
            if len(ckpt_file) == 1:
                cfg.ckpt_map = [cfg.ckpt_map]
            assert len(ckpt_file) == len(cfg.ckpt_map)
            [model.load(_, __) for _, __ in zip(ckpt_file, cfg.ckpt_map)]
        else:
            model.load(ckpt_file, cfg.ckpt_map)
    if cfg.freez:
        model.freez(cfg.freez)

    model = model.cuda()
    model.compile()
    memory_report("model_ready")
    if cfg.param_groups:
        cfg.optimiz.params = model.group_params(**cfg.param_groups)
    else:
        cfg.optimiz.params = model.parameters()
    optimiz = build_from_config(cfg.optimiz)
    optimiz.gscale = build_from_config(cfg.gscale)
    optimiz.gclip = build_from_config(cfg.gclip)

    loss_fn_t = MetricWrap(**build_from_config(cfg.loss_fn_t))
    loss_fn_v = MetricWrap(**build_from_config(cfg.loss_fn_v))
    acc_fn_t = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_t))
    acc_fn_v = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v))

    for cb in cfg.callback_t + cfg.callback_v:
        if cb.type.__name__ in ["AverageLog", "HandleLog"]:
            cb.log_file = f"{save_path}.txt"
        elif cb.type.__name__ == "SaveModel":
            cb.save_dir = save_path
    callback_t = build_from_config(cfg.callback_t)
    callback_v = build_from_config(cfg.callback_v)

    pack = Config({})
    pack.dataset_t = dataload_t
    pack.dataset_v = dataload_v
    pack.model = model
    pack.optimiz = optimiz
    pack.loss_fn_t = loss_fn_t
    pack.loss_fn_v = loss_fn_v
    pack.acc_fn_t = acc_fn_t
    pack.acc_fn_v = acc_fn_v
    pack.callback_t = callback_t
    pack.callback_v = callback_v
    pack.total_step = cfg.total_step
    pack.val_interval = cfg.val_interval
    pack.save_path = save_path

    epoch_count = 0
    epoch_count_v = 0
    pack.step_count = 0
    [_.before_train(**pack) for _ in pack.callback_t]

    while pack.step_count < pack.total_step:
        pack.epoch = epoch_count
        pt.cuda.empty_cache()
        train_epoch(pack)

        flag1 = pack.step_count >= (epoch_count_v + 1) * pack.val_interval
        flag2 = pack.step_count >= pack.total_step
        if flag1 or flag2:
            pt.cuda.empty_cache()
            val_epoch(pack)
            epoch_count_v += 1

        epoch_count += 1
        memory_report(f"epoch_{epoch_count}")

    assert pack.step_count == pack.total_step
    [_.after_train(**pack) for _ in pack.callback_t]

    completion = {
        "seed": seed,
        "config": str(cfg_file),
        "ranking_mode": ranking_mode,
        "ranking_alpha": ranking_alpha,
        "total_step": int(pack.total_step),
        "status": "complete",
    }
    (save_path / "TRAINING_COMPLETE.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    del pack, model, optimiz, dataload_t, dataload_v, dataset_t, dataset_v
    del loss_fn_t, loss_fn_v, acc_fn_t, acc_fn_v, callback_t, callback_v
    pt.cuda.empty_cache()
    gc.collect()
    print(f"===== Seed {seed} training completed =====\n")


def main(args):
    cfg_file = Path(args.cfg_file)
    data_path = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    ckpt_file = args.ckpt_file

    if ckpt_file is None:
        pass
    elif isinstance(ckpt_file, list) and len(ckpt_file) == 1:
        ckpt_file = Path(ckpt_file[0])
    else:
        ckpt_file = [Path(_) for _ in ckpt_file]

    assert cfg_file.name.endswith(".py")
    assert cfg_file.is_file()
    print(f"Waitting seeds: {args.seeds}")
    print(args)
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("num_workers 不能为负数")

    for seed in args.seeds:
        print(f"\n{'='*50}")
        print(f"Start Seed = {seed}")
        print(f"{'='*50}")
        run_single_seed(
            seed, cfg_file, data_path, save_dir, ckpt_file,
            args.ranking_mode, args.ranking_alpha, args.num_workers, args.low_memory,
        )

    print("\nALL seeds training completed!")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        # default=[43],
    )
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="config-smoothsa/smoothsa_r-voc_AGM.py",
        # default="config-smoothsa/smoothsav_c-movi_e_AGM.py",
        # default="config-smoothsa/smoothsav_r-ytvis_AGM.py",
    )
    parser.add_argument("--data_dir", type=str, default="../voc")
    parser.add_argument("--save_dir", type=str, default="save")
    parser.add_argument("--ranking_mode", choices=["guided", "random"], default="guided")
    parser.add_argument("--ranking_alpha", type=float, default=None,
                        help="覆盖 decoder 的先验混合系数 alpha；留空时使用配置值。")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="覆盖配置中的 DataLoader worker 数；0 为最低主机内存模式。")
    parser.add_argument("--low_memory", action="store_true",
                        help="关闭 pinned memory，并在多 worker 时使用 prefetch_factor=1。")
    parser.add_argument(
        "--ckpt_file",
        type=str,
        nargs="+",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
