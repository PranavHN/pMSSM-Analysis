#!/bin/bash
# Offline check suite: the unit tests (including the sampled-point golden file)
# and workflow DAG dry-runs in every mode. No physics tools, no cluster, no
# network — the whole thing takes seconds.
#
# This is the SINGLE definition of those checks, shared by `pixi run test` and
# the CI dry-run job, so what you run locally cannot drift from what CI runs.
#
# Usage:  pixi run test        (or, with the env already set up:
#         . build/setup.sh && bash tests/run_checks.sh)
#
# NB: a dry-run does not verify that a rule's command exists, so this suite
# cannot catch a broken path to a rule body — only a real scan does.
set -euo pipefail

# Locate the checkout from this script, so it works from any cwd and without
# the build environment having exported RUN3MODELGENDIR.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SMK=(snakemake -s workflow/Snakefile --workflow-profile workflow/profiles/local -n)
CFGD=source/Run3ModelGen/data/configs

echo "== unit tests (incl. the sampled-point golden file) =="
pytest tests -q

echo "== workflow DAG: single scan =="
"${SMK[@]}" --config mode=single scan_dir=run/dryrun seed=123

echo "== workflow DAG: batch fan-out + EOS publish =="
"${SMK[@]}" --config mode=batch runid=1 batches=3 scan_parent=run publish=eos

# Publishing is a MOVE. Two halves, both silent when wrong: the publish rule
# must be told to clean up, and the ntuples must NOT be requested targets —
# snakemake regenerates a missing *requested* file, so listing them would re-run
# every batch to re-create what publish just deleted. --keep_local inverts both.
# NB: dry-run output goes to a file, never into `grep -q` — the early exit
# SIGPIPEs snakemake and pipefail turns that into a spurious failure.
dry="$(mktemp)"
trap 'rm -f "$dry"' EXIT

# `rule all`'s input line = exactly the requested targets.
all_targets () { awk '/^(local)?rule all:/{f=1} f&&/^ +input:/{print; exit}' "$dry"; }

echo "== workflow DAG: publishing is a move (cleanup on, ntuples not targets) =="
"${SMK[@]}" -p --config mode=batch runid=1 batches=2 scan_parent=run publish=eos > "$dry"
if ! grep -q -- "r3mg_batch.py publish .*--cleanup" "$dry"; then
    echo "FAIL: the publish rule does not pass --cleanup by default"; exit 1
fi
case "$(all_targets)" in
    *ntuple*) echo "FAIL: ntuple is a requested target although publish deletes it"; exit 1 ;;
esac

echo "== workflow DAG: publish with keep_local keeps both =="
"${SMK[@]}" -p --config mode=batch runid=1 batches=2 scan_parent=run publish=eos keep_local=1 > "$dry"
if grep -q -- "--cleanup" "$dry"; then
    echo "FAIL: keep_local must not pass --cleanup"; exit 1
fi
case "$(all_targets)" in
    *ntuple*) : ;;
    *) echo "FAIL: keep_local must keep the ntuples as requested targets"; exit 1 ;;
esac

echo "== workflow DAG: mcmc chain =="
"${SMK[@]}" --config mode=single scan_dir=run/dryrun_mcmc config_file="$CFGD/ci_mcmc_config.yaml"

echo "== workflow DAG: extraction fan-out =="
"${SMK[@]}" --config mode=extract eos_dir=/fake/123 procids=0,1 work_dir=run

# The entrypoint ships the selection base64-encoded, because the raw awkward
# expression's quotes break the HTCondor submit description. Check the rule
# decodes it back into the extractModels.py command line (shell-quoted, so
# compare parsed tokens rather than raw text).
echo "== workflow DAG: extraction with a base64 selection =="
SEL="akarr['SS_m_h']!=-1"
SEL_B64=$(printf %s "$SEL" | base64 -w0)
"${SMK[@]}" -p --config mode=extract eos_dir=/fake/123 procids=0 work_dir=run \
    "selection_b64=$SEL_B64" \
    | SEL="$SEL" python3 -c '
import os, shlex, sys
want = os.environ["SEL"]
# read everything: exiting early would break snakemake pipe (pipefail)
lines = sys.stdin.read().splitlines()
for line in lines:
    if "extractModels.py" not in line:
        continue
    argv = shlex.split(line.split("Shell command:", 1)[-1])
    if want == argv[argv.index("--selection") + 1]:
        sys.exit(0)
sys.exit("FAIL: selection_b64 did not reach the shell command as " + want)
'

echo "All offline checks passed."
