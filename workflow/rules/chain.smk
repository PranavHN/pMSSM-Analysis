# prior: mcmc — the emcee ensemble sampler runs the WHOLE pipeline inside sampling
# (propose -> pipeline -> log-prob), so it cannot be decomposed into per-tool rules:
# one opaque chain job per batch, owning its scan dir. Engine dispatch is in
# r3mg_batch.py chain -> Run3ModelGen.emcee_sampler.run_emcee_scan. It checkpoints to
# chain.h5 and auto-resumes: chain.h5 + chain.h5.meta.json are UNDECLARED side-state
# in the scan dir, so Snakemake never deletes them and re-running the rule resumes the
# chain (it deliberately does NOT wipe the scan dir). As its FINAL step the chain also
# writes emcee diagnostics (R-hat/ESS/tau + corner/trace/log-prob plots) into
# <scan_dir>/diag — a declared directory output, produced best-effort so a plotting
# error never fails a completed chain.
#
# Pool width = cpus-per-task = threads: the sampler runs a worker Pool of
# `mcmc.nworkers`; wiring threads from the parsed config (parse-time pure, reads the
# CFG dict) makes the slurm/htcondor plugins request exactly that many cores (no
# oversubscription — BLAS is pinned to 1 in r3mg_batch.py). Absent nworkers -> 1.
_chain_threads = max(1, int(((CFG.get("mcmc") or {}).get("nworkers")) or 1))

# Memory scales with the worker Pool: each of the `nworkers` workers runs a full
# SPheno/micromegas/... pipeline concurrently (~2 GB peak each). A per-worker
# allotment lets the slurm/htcondor plugins request enough RAM for wide (e.g.
# 128-worker) emcee runs. Override per run with `--set-resources chain:mem_mb=<n>`;
# ignored by the local executor unless `--resources mem_mb=<n>` is also given.
_chain_mem_mb = _chain_threads * 2000

if MODE == "single":

    _sname = next(iter(SCANS))

    rule chain:
        input:
            warm = [f"{WARM}/pythia8.ok", f"{WARM}/smodels_db.ok"]
        output:
            ntuple = f"{SCANS[_sname]['dir']}/{_ntuple_basename(_sname)}",
            manifest = f"{WORK}/{_sname}/chain.json",
            diag = directory(f"{SCANS[_sname]['dir']}/diag")
        params:
            sdir = SCANS[_sname]["dir"],
            seed = SCANS[_sname]["seed"],
            cfg = CFG_OPT
        threads: _chain_threads
        resources:
            mem_mb = _chain_mem_mb
        shell:
            "{RULE_SCRIPTS}/r3mg_batch.py chain {params.cfg} --scan_dir {params.sdir} "
            "--seed {params.seed} --manifest_out {output.manifest} --nworkers {threads}"

else:

    rule chain:
        input:
            warm = [f"{WARM}/pythia8.ok", f"{WARM}/smodels_db.ok"]
        output:
            ntuple = f"{BATCH_PREFIX}.{{b}}/ntuple.{RUNID}.{{b}}.root",
            manifest = f"{WORK}/scan.{RUNID}.{{b}}/chain.json",
            diag = directory(f"{BATCH_PREFIX}.{{b}}/diag")
        params:
            sdir = lambda w: f"{BATCH_PREFIX}.{w.b}",
            seed = lambda w: SCANS[f"scan.{RUNID}.{w.b}"]["seed"],
            cfg = CFG_OPT
        threads: _chain_threads
        resources:
            mem_mb = _chain_mem_mb
        shell:
            "{RULE_SCRIPTS}/r3mg_batch.py chain {params.cfg} --scan_dir {params.sdir} "
            "--seed {params.seed} --manifest_out {output.manifest} --nworkers {threads}"
