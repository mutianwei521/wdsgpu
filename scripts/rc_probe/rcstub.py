"""Exit with the status named in argv[1]. Stands in for a measurement script
so the job's recorded Slurm state can be checked without burning GPU time."""
import sys
code = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print("rcstub: exiting %d" % code, flush=True)
sys.exit(code)
