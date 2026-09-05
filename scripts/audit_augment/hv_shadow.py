# -*- coding: utf-8 -*-
"""hv_shadow.py: bit-for-bit shadow test of dgga's default placement paths.

Mode --dump: run the pre-augmentation public functions of dgga.placement on the City D and Hanoi
sensitivity caches with the package found first on sys.path (pass --pkg <dir containing dgga>)
and dump every array to an npz.  Run it once with the shadow package (git archive of the
baseline commit) and once with the working tree, then --compare the two dumps: every array must
be identical bit for bit.  Mode --augment-identity (working tree only): bayes_dopt_augment with an
empty S0 must reproduce bayes_dopt_greedy bit for bit (order, gains, f curve, certificates).
"""
import argparse
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")


def flatten(prefix, obj, store):
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(f"{prefix}.{k}", v, store)
    elif isinstance(obj, (list, tuple)):
        store[prefix] = np.asarray(obj)
    elif isinstance(obj, np.ndarray):
        store[prefix] = obj
    elif isinstance(obj, (int, float, bool, np.integer, np.floating)):
        store[prefix] = np.asarray(obj)
    else:
        store[prefix] = np.asarray(str(obj))


def dump(pkg, out):
    sys.path.insert(0, pkg)
    import dgga.placement as pl
    print(f"  dgga.placement from {pl.__file__}")
    store = {}
    for stem, kmax, certs in (("city_d", 80, (5, 10, 20, 40, 80)), ("pub_hanoi", 10, (2, 4, 6, 8, 10))):
        z = np.load(os.path.join(DATA, f"placement_cache_{stem}.npz"))
        S = z["S_full"]
        g = pl.bayes_dopt_greedy(S, kmax, 15.0, 0.1, cert_ks=certs)
        g.pop("t_total", None)
        flatten(f"{stem}.greedy", g, store)
        order = np.asarray(g["order"])
        for k in certs:
            ev = pl.eval_subset(S, order[:k], 15.0, 0.1)
            flatten(f"{stem}.eval{k}", ev, store)
        flatten(f"{stem}.eval_pos40", pl.eval_subset(S, z["pos40"], 15.0, 0.1), store)
        zc, nz = pl.zero_col_curve(S, order.tolist())
        store[f"{stem}.zero_col_curve"] = zc
        store[f"{stem}.zero_col_mask"] = nz
        store[f"{stem}.random_orders"] = np.asarray(pl.random_orders(S.shape[1], n_seeds=5, seed0=0))
        store[f"{stem}.norm_order"] = np.asarray(pl.norm_order(S))
        ao = pl.aopt_greedy(S, 6, 15.0, 0.1)
        if isinstance(ao, dict):
            ao.pop("t_total", None)
        flatten(f"{stem}.aopt", ao, store)
        sub = pl.submodularity_check(S, 15.0, 0.1, n_samples=40, seed=0)
        if isinstance(sub, dict):
            sub.pop("t_total", None)
        flatten(f"{stem}.submod", sub, store)
    np.savez(out, **store)
    print(f"  dumped {len(store)} arrays to {out}")


def compare(a, b):
    A, B = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    ka, kb = set(A.files), set(B.files)
    bad = []
    if ka != kb:
        bad.append(f"key sets differ: only_a={sorted(ka - kb)[:5]} only_b={sorted(kb - ka)[:5]}")
    for k in sorted(ka & kb):
        x, y = A[k], B[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            bad.append(f"{k}: shape/dtype {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}")
            continue
        if x.dtype.kind in "fc":
            same = np.array_equal(x.view(np.uint8) if x.ndim else x, y.view(np.uint8) if y.ndim else y) \
                if x.ndim else (x.tobytes() == y.tobytes())
        else:
            same = np.array_equal(x, y)
        if not same:
            bad.append(f"{k}: differs (max|d|={np.max(np.abs(x.astype(float) - y.astype(float))) if x.dtype.kind in 'fciu' else 'n/a'})")
    print(f"  compared {len(ka & kb)} arrays: {'ALL IDENTICAL BIT FOR BIT' if not bad else 'DIFFERENCES'}")
    for m in bad:
        print("   ", m)
    return not bad


def augment_identity():
    sys.path.insert(0, ROOT)
    import dgga.placement as pl
    ok = True
    for stem, kmax, certs in (("city_d", 80, (5, 10, 20, 40, 80)), ("pub_hanoi", 10, (2, 4, 6, 8, 10))):
        S = np.load(os.path.join(DATA, f"placement_cache_{stem}.npz"))["S_full"]
        g = pl.bayes_dopt_greedy(S, kmax, 15.0, 0.1, cert_ks=certs)
        a = pl.bayes_dopt_augment(S, [], kmax, 15.0, 0.1, objective="dopt", cert_ks=certs)
        same_order = g["order"] == a["order"]
        same_gains = np.asarray(g["gains"]).tobytes() == np.asarray(a["gains"]).tobytes()
        same_f = np.asarray(g["f_curve"]).tobytes() == np.asarray(a["f_curve"][1:]).tobytes()
        f0_zero = a["f0"] == 0.0
        cert_same = all(g["cert"][k]["f"] == a["cert"][k]["f"] and g["cert"][k]["upper"] == a["cert"][k]["upper_df"]
                        and g["cert"][k]["ratio"] == a["cert"][k]["ratio"] for k in certs)
        print(f"  [{stem}] augment(S0=[]) vs greedy: order {same_order} gains {same_gains} f_curve {same_f} "
              f"f0==0 {f0_zero} certs {cert_same} n_evals {g['n_evals']}/{a['n_evals']}")
        ok &= same_order and same_gains and same_f and f0_zero and cert_same
    print(f"  augment identity: {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="output npz")
    ap.add_argument("--pkg", default=ROOT, help="directory whose dgga/ package is used")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--augment-identity", action="store_true")
    a = ap.parse_args()
    rc = 0
    if a.dump:
        dump(a.pkg, a.dump)
    if a.compare:
        rc |= 0 if compare(*a.compare) else 1
    if a.augment_identity:
        rc |= 0 if augment_identity() else 2
    sys.exit(rc)
