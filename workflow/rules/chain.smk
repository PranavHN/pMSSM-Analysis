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

    # ── pooled cross-ensemble diagnostics ───────────────────────────────────
    # Each chain job diagnoses only ITS OWN backend, so every per-batch diag/ is
    # single-ensemble and labels everything ens0. The authoritative numbers -- the
    # raw / mu>=0-canonical / magnitude-fold R-hat blocks, the cross-seed rank
    # statistic, the octant-occupancy spread and the pooled corner -- need all the
    # backends at once. Within-ensemble R-hat must NOT be read as the verdict: the
    # DE/snooker moves couple walkers, so it is only a lower bound (v3's
    # within-ensemble canonical R-hat was 1.173 against v2's cross-ensemble ~1.9).
    #
    # Inputs are the NTUPLES, not the chain.h5 files: the backends are undeclared
    # side-state with no producing rule, so listing them would fail at DAG build.
    # The ntuples span the same DAG, and since resuming a batch rewrites its ntuple,
    # a resume re-triggers this automatically.
    #
    # Deliberately NOT in rule all (see _final_targets in the Snakefile): a
    # plotting/arviz hiccup must not fail an otherwise-complete production run,
    # exactly as for the per-batch diagnostics. Ask for it by name:
    #     pixi run smk ... <meta_parent>/r3mg/diag.<runid>
    if BATCHES > 1:

        rule chain_diag_all:
            input:
                ntuples = expand(f"{BATCH_PREFIX}.{{b}}/ntuple.{RUNID}.{{b}}.root",
                                 b=range(BATCHES))
            output:
                diag = directory(f"{META}/diag.{RUNID}")
            params:
                pattern = f"{BATCH_PREFIX}.*/chain.h5"
            # Peak is the unthinned pooled arviz object, ~ nseeds * nwalkers *
            # (min_iter - discard) * ndim * 8 bytes, and each fold copies it (1.24 GB
            # apiece for a 4 x 256 x 24k run, which took ~13 min). 32 GB leaves room;
            # the chain rule's own allotment is sized for the sampler pool instead.
            resources:
                mem_mb = 32000,
                runtime = 90
            shell:
                # Guard, not decoration: under htcondor scratch scan dirs, or a
                # --publish without --keep_local, the backends are scattered across
                # workers or already removed, and the glob would quietly match one
                # file (or none) -- yielding a "cross-ensemble" report built from a
                # single chain, which is exactly the misreading this rule exists to
                # prevent.
                "n=$(ls -1 {params.pattern} 2>/dev/null | wc -l); "
                'if [ "$n" -lt 2 ]; then '
                '  echo "chain_diag_all: $n backend(s) match {params.pattern}, need >= 2." >&2; '
                '  echo "  All scan dirs must be readable from here: not the case for" >&2; '
                '  echo "  htcondor scratch dirs, or --publish without --keep_local." >&2; '
                "  exit 1; "
                "fi; "
                "emcee_diagnostics.py --glob '{params.pattern}' --outdir {output.diag}"
