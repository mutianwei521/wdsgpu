#!/bin/bash
# 本轮审阅两个作业脚本的 rc 传递自查（判据与 scripts/rc_probe/check_rc_local.sh 同款）：
# 用 stub 的 python3/nvidia-smi 替身跑一遍，比对脚本退出码与期望。
set -u
SB=$(mktemp -d); trap 'rm -rf "$SB"' EXIT; mkdir -p "$SB/bin"
cat > "$SB/bin/nvidia-smi" <<'EOS'
#!/bin/bash
echo "stub-gpu, 0 MiB, 0.0"
EOS
cat > "$SB/bin/python3" <<'EOS'
#!/bin/bash
target=""
for a in "$@"; do case "$a" in *.py) target="$a"; break;; esac; done
rc=0
[ -f "$RCMAP" ] && rc=$(awk -v t="$target" '$1==t {print $2}' "$RCMAP")
[ -z "$rc" ] && rc=0
echo "stub python3 $target -> exit $rc"
exit $rc
EOS
cp "$SB/bin/python3" "$SB/bin/python"; chmod +x "$SB/bin"/*
command -v cygpath >/dev/null 2>&1 && SB=$(cygpath -u "$SB")
export PATH="$SB/bin:$PATH" SLURM_SUBMIT_DIR="$SB" RCMAP="$SB/rcmap"
bad=0
run() { local s="$1"; shift; : > "$RCMAP"; for l in "$@"; do echo "$l" >> "$RCMAP"; done
        bash "$s" >/dev/null 2>&1; echo $?; }
chk() { local d="$1" e="$2" g="$3" v=ok
        [ "$e" = "$g" ] || { v="** MISMATCH **"; bad=1; }
        printf "%-46s exp=%-2s got=%-2s %s\n" "$d" "$e" "$g" "$v"; }
D=scripts/p5_bignet/audit
chk "aud2.sh python ok"        0 "$(run $D/aud2.sh 'aud_gpu_a2.py 0')"
chk "aud2.sh python exits 3"   3 "$(run $D/aud2.sh 'aud_gpu_a2.py 3')"
chk "aud2.sh python exits 9"   9 "$(run $D/aud2.sh 'aud_gpu_a2.py 9')"
chk "aud3.sh python ok"        0 "$(run $D/aud3.sh 'aud_gpu_a3.py 0')"
chk "aud3.sh python exits 5"   5 "$(run $D/aud3.sh 'aud_gpu_a3.py 5')"
chk "aud3.sh python exits 2"   2 "$(run $D/aud3.sh 'aud_gpu_a3.py 2')"
chk "aud4.sh python ok"        0 "$(run $D/aud4.sh 'aud_gpu_a4.py 0')"
chk "aud4.sh python exits 4"   4 "$(run $D/aud4.sh 'aud_gpu_a4.py 4')"
echo
[ $bad -eq 0 ] && echo "RC SELF-CHECK: 8/8 OK" || echo "RC SELF-CHECK: FAILED"
exit $bad
