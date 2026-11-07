# -*- coding: utf-8 -*-
"""gpu_transfer_timing.py - RTX 5090 上的数据传输耗时（WR 大修 R2-M6，2026-10-23）。

与 mainline_v2/mv2_time_gpu.py 的"新c"列同一调用：implicit_solve(adjoint='gpu', csr 装配 + cuDSS，
批量状态机，accuracy=1e-6)，对需水求梯度，损失 L = Σ W·H_junction。每个网络、每个批量 B 分别计时：
  up        主机 numpy（可分页内存）→ 显存：需水 D、定水头 R、权重 W，三个 [B,N] 或 [B,Nj] 的 float64 数组
  up_pinned 锁页内存 → 显存（non_blocking；不含锁页本身的时间），作为最好情况
  fb        数据已在显存时的一次前向加反向（结果留在显存）
  down      水头 [B,N] 与需水梯度 [B,N] 取回主机
  e2e       从主机数组到主机结果的完整一次调用 = up + fb + down
  内部拷贝  用 torch.profiler 统计一次 fb 调用里的 CUDA memcpy（HtoD / DtoH / DtoD）次数与设备耗时；
            fb 里有一次为求导复制输入的 DtoD，是计时脚手架本身产生的。
计时：预热 1 次后取中位数（B<=256 取 5 次，B=1024 取 3 次），每次前后同步。
输出：transfer_timing.json（与脚本同目录），stdout 表格。
运行（5090）：sbatch --gpus=1 -p gpu_5090 -t 00:40:00 --wrap "python3 -X utf8 gpu_transfer_timing.py"
"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(os.path.join(ROOT, "dgga")):            # 本机布局：scripts/ 的上一级
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8")
from dgga.autodiff import implicit_solve                     # noqa: E402
from dgga.parse import parse_inp                             # noqa: E402
from dgga.solver import GGASolver                            # noqa: E402

DEV, DT, ACC = "cuda", torch.float64, 1e-6
NETS = {"L-TOWN": "networks/public/_cleaned/L-TOWN.inp", "City D": "networks/realInpData/city_d.inp"}
BS = [1, 64, 256, 1024]


def sync():
    torch.cuda.synchronize()


def tmed(fn, reps):
    fn()
    sync()
    ts = []
    for _ in range(reps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def boundary(net):
    """与 mv2_time_gpu.boundary 相同：t=0 需水；水池水头取 30%–70% 区间的中点。"""
    d = np.asarray(net.demand_cfs_at(0), dtype=np.float64)
    rh = np.array(net.reservoir_head_ft_at(0), dtype=np.float64)
    tn = np.asarray(net.tank_node, dtype=np.int64)
    if tn.size:
        lo = net.tank_hmin + 0.30 * (net.tank_hmax - net.tank_hmin)
        hi = net.tank_hmax - 0.30 * (net.tank_hmax - net.tank_hmin)
        rh[tn] = np.clip(0.5 * (net.tank_hmin + net.tank_hmax), lo, hi)
    return d, np.nan_to_num(rh)


def memcpy_profile(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        sync()
    out = {}
    for e in prof.key_averages():
        k = e.key
        if "emcpy" not in k:
            continue
        dt = getattr(e, "device_time_total", None)
        if dt is None:
            dt = getattr(e, "cuda_time_total", 0.0)
        tag = ("HtoD" if "HtoD" in k else "DtoH" if "DtoH" in k else "DtoD" if "DtoD" in k
               else "runtime:" + k[:24])
        c, t = out.get(tag, (0, 0.0))
        out[tag] = (c + int(e.count), t + float(dt) / 1e3)       # 微秒 → 毫秒
    return out


def main():
    print(f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda} "
          f"node={os.popen('hostname').read().strip()} ACC={ACC}")
    allr = {}
    for name, rel in NETS.items():
        inp = os.path.join(ROOT, rel)
        net = parse_inp(inp)
        s = GGASolver(net, device=DEV, dtype=DT, mode="dense", inp_path=inp, dense_status_machine=True)
        jn = torch.as_tensor(np.asarray(s.junc_nodes), dtype=torch.long, device=DEV)
        d0, rh0 = boundary(net)
        print(f"\n### {name}  N={net.N} Nj={s.Nj} L={s.L}")
        print(f"{'B':>5} | {'上传':>9} {'上传(锁页)':>10} | {'前向+反向':>10} | {'取回':>9} | {'端到端':>10} | "
              f"{'传输占比':>7} | {'每工况 端到端/驻留':>18} | 一次前向+反向内部的拷贝（次数, 设备 ms）")
        for B in BS:
            g = np.random.default_rng(2026)
            Dn = d0[None, :] * g.uniform(0.85, 1.15, (B, d0.size))
            Rn = rh0[None, :] + g.uniform(-1.0, 1.0, (B, rh0.size))
            Wn = g.normal(size=(B, s.Nj))

            def up():
                return (torch.as_tensor(Dn, dtype=DT, device=DEV), torch.as_tensor(Rn, dtype=DT, device=DEV),
                        torch.as_tensor(Wn, dtype=DT, device=DEV))
            D, R, W = up()

            def fb(D=D, R=R, W=W):
                Dv = D.detach().clone().requires_grad_(True)
                h, _q, _e = implicit_solve(s, Dv, R, adjoint="gpu", accuracy=ACC, max_iter=200,
                                           status_machine=True, assemble="csr", linear_solver="cudss")
                (h.index_select(1, jn) * W).sum().backward()
                return h, Dv.grad
            h, gD = fb()

            def down(h=h, gD=gD):
                return h.detach().cpu().numpy(), gD.cpu().numpy()

            def e2e():
                D_, R_, W_ = up()
                h_, g_ = fb(D_, R_, W_)
                return h_.detach().cpu().numpy(), g_.cpu().numpy()
            reps = 5 if B <= 256 else 3
            t_up, t_fb, t_dn, t_e2e = tmed(up, reps), tmed(fb, reps), tmed(down, reps), tmed(e2e, reps)
            Dp, Rp, Wp = (torch.from_numpy(Dn).pin_memory(), torch.from_numpy(Rn).pin_memory(),
                          torch.from_numpy(Wn).pin_memory())
            t_up_pin = tmed(lambda: (Dp.to(DEV, non_blocking=True), Rp.to(DEV, non_blocking=True),
                                     Wp.to(DEV, non_blocking=True)), reps)
            mc = memcpy_profile(fb)
            frac = (t_up + t_dn) / t_e2e
            rec = dict(B=B, up_s=t_up, up_pinned_s=t_up_pin, fb_resident_s=t_fb, down_s=t_dn, e2e_s=t_e2e,
                       transfer_fraction_of_e2e=frac, bytes_up=int(Dn.nbytes + Rn.nbytes + Wn.nbytes),
                       bytes_down=int(h.numel() * 8 + gD.numel() * 8),
                       per_scenario_e2e_ms=t_e2e / B * 1e3, per_scenario_resident_ms=t_fb / B * 1e3,
                       internal_memcpy={k: dict(count=v[0], device_ms=v[1]) for k, v in mc.items()})
            allr[f"{name}|{B}"] = rec
            mcs = ", ".join(f"{k} {v[0]}次 {v[1]:.3f}ms" for k, v in sorted(mc.items()))
            print(f"{B:>5} | {t_up*1e3:7.3f}ms {t_up_pin*1e3:8.3f}ms | {t_fb*1e3:8.2f}ms | {t_dn*1e3:7.3f}ms | "
                  f"{t_e2e*1e3:8.2f}ms | {frac*100:6.2f}% | {t_e2e/B*1e3:8.3f} / {t_fb/B*1e3:7.3f} ms | {mcs}",
                  flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transfer_timing.json")
    json.dump(allr, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\nsaved -> {path}\nTRANSFER DONE")


if __name__ == "__main__":
    main()
