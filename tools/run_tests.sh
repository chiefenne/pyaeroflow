#!/usr/bin/env bash
# Run the test suite with one pytest process per file, so memory is freed
# between files (the full suite in one process can run out of memory).
#
# Usage:
#   tools/run_tests.sh                 # default suite (unit + verification, not slow)
#   tools/run_tests.sh --slow          # slow verification tests only
#   tools/run_tests.sh --jobs 2        # run 2 files in parallel (default 1)
#   tools/run_tests.sh --out DIR       # output folder (default test_runs/<timestamp>)
#   tools/run_tests.sh FILE [FILE...]  # only these test files
#
# Writes DIR/<file>.log per test file, DIR/lint.log for ruff and
# DIR/summary.txt with pass/fail counts and every failing test.

set -u
MODE=default
JOBS=1
OUT=""
FILES=()

while [ $# -gt 0 ]; do
    case "$1" in
        --slow) MODE=slow ;;
        --jobs) JOBS="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) FILES+=("$1") ;;
    esac
    shift
done

cd "$(dirname "$0")/.." || exit 1
if [ ${#FILES[@]} -eq 0 ]; then
    FILES=(tests/unit/test_*.py tests/verification/test_*.py)
fi
[ -z "$OUT" ] && OUT="test_runs/$(date +%Y%m%d_%H%M%S)_$MODE"
mkdir -p "$OUT"

run_one() {
    # exported to xargs' bash, so the mode is passed by variable, not array
    local f="$1" name log rc mark=()
    [ "$MARK_MODE" = slow ] && mark=(-m slow)
    name=$(echo "$f" | tr '/' '_' | sed 's/\.py$//')
    log="$OUT/$name.log"
    # -v prints each test with a running percentage, e.g. "[ 3%]"
    python -m pytest -v -p no:cacheprovider -rfE "${mark[@]}" "$f" > "$log" 2>&1
    rc=$?
    echo "$rc $f" >> "$OUT/status.txt"
    # one line per finished file: position, file, last summary line
    n=$(wc -l < "$OUT/status.txt")
    echo "[$n/$TOTAL] $(date +%H:%M:%S) $f: $(grep -E '[0-9]+ (passed|failed|error|skipped)' "$log" | tail -n 1 | tr -d '=' | sed 's/^ *//')"
}
export -f run_one
export OUT MARK_MODE="$MODE" TOTAL="${#FILES[@]}"
: > "$OUT/status.txt"

echo "Running ${#FILES[@]} files ($MODE, $JOBS at a time) -> $OUT"
printf '%s\n' "${FILES[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

# Lint, same checks as CI-style review.
{
    echo "== ruff check"; ruff check . 2>&1 | tail -n 3
    echo "== ruff format --check"; ruff format --check . 2>&1 | tail -n 2
} > "$OUT/lint.log" 2>&1

# Summary: counts per file, totals and failing tests.
python - "$OUT" <<'PY'
import re, sys, pathlib
out = pathlib.Path(sys.argv[1])
rows, totals = [], {}
for line in (out / "status.txt").read_text().splitlines():
    rc, f = line.split(" ", 1)
    log = out / (f.replace("/", "_").removesuffix(".py") + ".log")
    text = log.read_text(errors="replace")
    last = [l for l in text.splitlines() if re.search(r"\d+ (passed|failed|skipped|error|deselected)", l)]
    counts = dict((k, int(n)) for n, k in re.findall(r"(\d+) (passed|failed|skipped|errors?|deselected)", last[-1])) if last else {}
    for k, v in counts.items():
        totals[k] = totals.get(k, 0) + v
    rows.append((rc, f, counts, [l for l in text.splitlines() if l.startswith(("FAILED", "ERROR"))]))
with open(out / "summary.txt", "w") as s:
    s.write("exit  file                                              passed failed skipped\n")
    for rc, f, c, _ in rows:
        s.write(f"{rc:>4}  {f:<50}{c.get('passed', 0):>6}{c.get('failed', 0):>7}{c.get('skipped', 0):>8}\n")
    s.write("\nTOTAL " + ", ".join(f"{k} {v}" for k, v in sorted(totals.items())) + "\n")
    bad = [(f, l) for _, f, _, ls in rows for l in ls]
    s.write("\nFAILING TESTS\n" if bad else "\nNo failing tests.\n")
    for f, l in bad:
        s.write(f"  {l}\n")
print((out / "summary.txt").read_text())
PY
