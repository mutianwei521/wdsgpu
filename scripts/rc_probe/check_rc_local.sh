#!/bin/bash
# Local check that every job script propagates its Python exit status.
# Runs each script against stub `python3`/`nvidia-smi` so nothing real is
# executed, and compares the script's exit code with what is expected.
# Not an sbatch job -- run it from the repo root:  bash scripts/rc_probe/check_rc_local.sh
# The cluster half of the evidence (Slurm State/ExitCode) lives in
# data/gpu/rc{ok,new,acc,old}_14600*.out; see data/rc_propagation_wip.txt.
set -u
SB=$(mktemp -d)
trap 'rm -rf "$SB"' EXIT
mkdir -p "$SB/bin"

cat > "$SB/bin/nvidia-smi" <<'EOS'
#!/bin/bash
echo "stub-gpu, 0 MiB"
EOS
cat > "$SB/bin/python3" <<'EOS'
#!/bin/bash
# Exit with the code RCMAP gives for this .py, default 0.
target=""
for a in "$@"; do case "$a" in *.py) target="$a"; break;; esac; done
rc=0
[ -f "$RCMAP" ] && rc=$(awk -v t="$target" '$1==t {print $2}' "$RCMAP")
[ -z "$rc" ] && rc=0
echo "stub python3 $target -> exit $rc"
exit $rc
EOS
cp "$SB/bin/python3" "$SB/bin/python"
chmod +x "$SB/bin"/*
# A Windows-style "C:/..." entry would split on its own colon, so use the
# POSIX form when one is available (git-bash / msys).
command -v cygpath >/dev/null 2>&1 && SB=$(cygpath -u "$SB")
export PATH="$SB/bin:$PATH" SLURM_SUBMIT_DIR="$SB" RCMAP="$SB/rcmap"

bad=0
run() { local s="$1"; shift; : > "$RCMAP"
        for l in "$@"; do echo "$l" >> "$RCMAP"; done
        bash "$s" >/dev/null 2>&1; echo $?; }
chk() { local d="$1" e="$2" g="$3" v=ok
        [ "$e" = "$g" ] || { v="** MISMATCH **"; bad=1; }
        printf "%-52s exp=%-2s got=%-2s %s\n" "$d" "$e" "$g" "$v"; }

chk "single p3a.sh, python ok"                0 "$(run scripts/p3_autograd/p3a.sh 'p3a_gpu.py 0')"
chk "single p3a.sh, python exits 3"           3 "$(run scripts/p3_autograd/p3a.sh 'p3a_gpu.py 3')"
chk "single p2.sh (was a bare python)"        7 "$(run scripts/p2_cudss/p2.sh 'p2.py 7')"
chk "single p4r7.sh (VAR= prefix)"            5 "$(run scripts/p4_remeasure/p4r7.sh 'p4r_r7_gpu.py 5')"
chk "single audr2.sh"                         2 "$(run scripts/aud_p4/audr2.sh 'aud_r2_gpu.py 2')"
chk "single regression_gpu.sh"                1 "$(run scripts/regression_gpu.sh 'regression_gpu.py 1')"
chk "accum adv2.sh, both steps ok"            0 "$(run scripts/audit_p2_adversarial/adv2.sh 'adv2_timing.py 0' 'p2.py 0')"
chk "accum adv2.sh, FIRST step fails"         4 "$(run scripts/audit_p2_adversarial/adv2.sh 'adv2_timing.py 4' 'p2.py 0')"
chk "accum adv2.sh, LAST step fails"          9 "$(run scripts/audit_p2_adversarial/adv2.sh 'adv2_timing.py 0' 'p2.py 9')"
chk "accum adv.sh, one loop cell fails"       2 "$(run scripts/audit_p2_adversarial/adv.sh 'adv_gpu.py 0' 'adv_mem.py 2')"
chk "accum adv.sh, all ok"                    0 "$(run scripts/audit_p2_adversarial/adv.sh 'adv_gpu.py 0' 'adv_mem.py 0')"
chk "accum f4b.sh, run() helper fails"        6 "$(run scripts/p2_f4/f4b.sh 'f4b_gpu.py 6')"
chk "accum audtbl.sh, spotcheck cell fails"   8 "$(run scripts/aud_p4/audtbl.sh 'aud_tbl_gpu.py 0' 'aud_mem_gpu.py 8')"
chk "accum run_bench.sh, FIRST of 2 fails"    4 "$(run deploy/paracloud/run_bench.sh 'gpu_bench.py 4' 'epanet_ref_bench.py 0')"
chk "CONTROL xf_rc_old.sh stays broken"       0 "$(run scripts/xfinal/rc/xf_rc_old.sh 'regression_gpu.py 3')"
chk "CONTROL rc_probe/rc_old.sh stays broken" 0 "$(run scripts/rc_probe/rc_old.sh 'rcstub.py 3')"

echo
if [ "$bad" -eq 0 ]; then echo "RC PROPAGATION: all cases OK"; else echo "RC PROPAGATION: FAILURES ABOVE"; fi
exit $bad
