# prior: mcmc — the Metropolis-Hastings chain runs the WHOLE pipeline inside
# sampling (proposal -> pipeline -> accept/reject), so it cannot be decomposed
# into per-tool rules: one opaque, sequential chain job per batch, owning its
# entire scan dir (including the legacy wipe-on-start).
#
# This rule is also the emcee seam: the future emcee engine keeps the exact
# same rule shape (config in, ntuple out) with engine dispatch inside
# r3mg_batch.py chain — plus chain.h5/meta.json side-state and --resume.

if MODE == "single":

    _sname = next(iter(SCANS))

    rule chain:
        input:
            warm = [f"{WARM}/pythia8.ok", f"{WARM}/smodels_db.ok"]
        output:
            ntuple = f"{SCANS[_sname]['dir']}/{_ntuple_basename(_sname)}",
            manifest = f"{WORK}/{_sname}/chain.json"
        params:
            sdir = SCANS[_sname]["dir"],
            seed = SCANS[_sname]["seed"],
            cfg = CFG_OPT
        shell:
            "{RULE_SCRIPTS}/r3mg_batch.py chain {params.cfg} --scan_dir {params.sdir} "
            "--seed {params.seed} --manifest_out {output.manifest}"

else:

    rule chain:
        input:
            warm = [f"{WARM}/pythia8.ok", f"{WARM}/smodels_db.ok"]
        output:
            ntuple = f"{BATCH_PREFIX}.{{b}}/ntuple.{RUNID}.{{b}}.root",
            manifest = f"{WORK}/scan.{RUNID}.{{b}}/chain.json"
        params:
            sdir = lambda w: f"{BATCH_PREFIX}.{w.b}",
            seed = lambda w: SCANS[f"scan.{RUNID}.{w.b}"]["seed"],
            cfg = CFG_OPT
        shell:
            "{RULE_SCRIPTS}/r3mg_batch.py chain {params.cfg} --scan_dir {params.sdir} "
            "--seed {params.seed} --manifest_out {output.manifest}"
