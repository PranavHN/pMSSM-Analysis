# prior: active_learning — the surrogate-guided loop (train -> score candidates -> select
# -> label with the pipeline -> repeat) runs as ONE opaque job per scan dir, like the
# emcee chain: every iteration labels its batch inside the loop, so it cannot be
# decomposed into per-tool rules. Body: r3mg_batch.py active_learning ->
# Run3ModelGen.active_learning.loop.run_active_learning_scan.
#
# Side-state in the scan dir (al_state.pt, iteration_NNN/) is UNDECLARED, so Snakemake
# never deletes it and re-running the rule RESUMES after the last finished iteration.
# The declared outputs (the scan's ntuple, the manifest, diag/) are produced only when
# every configured iteration is done; a wall-clock stop exits non-zero without them, so
# a retry or a re-run of the same command continues the loop.
#
# Resources: threads = active_learning.nworkers = the CPU labelling workers (each runs
# its own SPheno/micromegas/... pipeline, ~2 GB; with SModelS in the steps ~3.5 GB), plus
# the parent process that holds the surrogate on the GPU. The GPU itself is requested in
# the cluster profiles (slurm: the `gres: gpu:1` resource on the `active_learning` rule). Override per
# run with `--set-resources active_learning:mem_mb=<n>`.
#
# This file is parse-time pure (stdlib + yaml only; see the Snakefile docstring).
_al_threads = max(1, int(((CFG.get("active_learning") or {}).get("nworkers")) or 1))
_al_mem_mb = _al_threads * 2000 + 8000

if MODE == "single":

    _sname = next(iter(SCANS))

    rule active_learning:
        input:
            warm = [f"{WARM}/pythia8.ok", f"{WARM}/smodels_db.ok"]
        output:
            ntuple = f"{SCANS[_sname]['dir']}/{_ntuple_basename(_sname)}",
            manifest = f"{WORK}/{_sname}/active_learning.json",
            diag = directory(f"{SCANS[_sname]['dir']}/diag")
        params:
            sdir = SCANS[_sname]["dir"],
            seed = SCANS[_sname]["seed"],
            cfg = CFG_OPT
        threads: _al_threads
        resources:
            mem_mb = _al_mem_mb
        shell:
            "{RULE_SCRIPTS}/r3mg_batch.py active_learning {params.cfg} --scan_dir {params.sdir} "
            "--seed {params.seed} --manifest_out {output.manifest} --nworkers {threads}"

else:

    rule active_learning:
        input:
            warm = [f"{WARM}/pythia8.ok", f"{WARM}/smodels_db.ok"]
        output:
            ntuple = f"{BATCH_PREFIX}.{{b}}/ntuple.{RUNID}.{{b}}.root",
            manifest = f"{WORK}/scan.{RUNID}.{{b}}/active_learning.json",
            diag = directory(f"{BATCH_PREFIX}.{{b}}/diag")
        params:
            sdir = lambda w: f"{BATCH_PREFIX}.{w.b}",
            seed = lambda w: SCANS[f"scan.{RUNID}.{w.b}"]["seed"],
            cfg = CFG_OPT,
            batch = lambda w: int(w.b)
        threads: _al_threads
        resources:
            mem_mb = _al_mem_mb
        shell:
            "{RULE_SCRIPTS}/r3mg_batch.py active_learning {params.cfg} --scan_dir {params.sdir} "
            "--seed {params.seed} --manifest_out {output.manifest} --nworkers {threads} "
            "--batch_index {params.batch}"
