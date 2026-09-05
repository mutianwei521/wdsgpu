Verbatim stdout of `python scripts/bench_batch.py`, run 2026-08-09 during the
repository-packaging task. These are the numbers quoted in the README's
"Batch and GPU performance" table.

An independent run of the same script is recorded in data/bench_batch_report.txt;
timings there differ by up to ~10% (run-to-run jitter on a laptop GPU/CPU),
while the accuracy figures (batch-vs-loop 0.000e+00, float32 drift 1.266e+02 ft)
are identical.

Interpreter: <python>
--------------------------------------------------------------------------------
=== bench_batch: city_d ===
N=542 Nj=541 L=554  场景: 需水×U(0.8,1.2) seed=20260808  torch 2.13.0+cu132  threads=24
[一致性] dense 批量(B=64) vs 逐场景循环：
  max|ΔH| = 0.000e+00 ft (门槛 1e-12)  max|ΔQ| = 0.000e+00 cfs  迭代数全等 = True  位级一致 = True
  判定: PASS
  批量迭代数分布: min=4 max=5
  我们 dense CPU float64 B=64: 迭代数 min=4 max=5
[性能] 我们 dense CPU float64: B=1: 8.501 ms/场景  B=64: 14.248 ms/场景  B=256: 17.538 ms/场景
GPU: NVIDIA GeForce RTX 5060 Laptop GPU
  我们 dense GPU float64 B=64: 迭代数 min=4 max=5
[性能] 我们 dense GPU float64: B=1: 21.885 ms/场景  B=64: 2.534 ms/场景  B=256: 2.180 ms/场景
  GPU f64 vs CPU f64 (B=64): max|ΔH| = 2.515e-05 ft
  我们 dense GPU float32 B=64: 迭代数 min=15 max=24
[性能] 我们 dense GPU float32: B=1: 60.640 ms/场景  B=64: 6.500 ms/场景  B=256: 4.603 ms/场景
  float32 精度观察 (B=64): max|ΔH| vs float64 = 1.266e+02 ft
[性能] 我们 epanet 模式 CPU (B=64): 6.768 ms/场景
[性能] EPANET (B=64, 冷启动): 设需水 0.466 + 求解 0.171 + 读头 0.384 = 1.021 ms/场景  (运行警告 0 场景)
[交叉] EPANET vs 我们epanet模式 (64场景 junction): max|ΔH|=3.195e-06 ft （需水缩放结合律 1ulp 差 × κ 放大，观察值）
[交叉] EPANET vs 我们dense CPU f64: max|ΔH|=1.137e-05 ft

===== 汇总（每场景 ms，city_d，冷启动稳态）=====
实现                                 B=1        B=64       B=256
我们 dense CPU f64                 8.501      14.248      17.538
我们 dense GPU f64                21.885       2.534       2.180
我们 dense GPU f32                60.640       6.500       4.603
我们 epanet 模式 CPU                     -       6.768           -
EPANET DLL(仅求解)                      -       0.171           -
EPANET DLL(含设需水+读头)                  -       1.021           -
float32 观察: max|ΔH| vs float64 = 1.266e+02 ft
一致性判定: PASS (max|ΔH| 批量 vs 逐场景 < 1e-12 ft)
