# Point-sampled priors (flat/fixed/grid/importance): sample once per batch,
# then one rule instance per (tool, batch), linearly chained in config-step
# order through the per-step manifests — reproducing the legacy list-order
# failure gating exactly (see modelgen.generate_models), so per-step success
# counts match the CI references. Per-model files inside the scan dir are NOT
# declared outputs (they legitimately don't exist for failed models); the
# manifests in META are the declared, atomic rule products.


def _step_inputs(wildcards):
    i = STEP_NAMES.index(wildcards.step)
    ins = {"points": f"{META}/{wildcards.scan}/points.npz"}
    if i > 0:
        ins["prev"] = f"{META}/{wildcards.scan}/step_{STEP_NAMES[i-1]}.json"
    if wildcards.step == "SModelS":
        # in-process smodels needs the built pythia8 + prefetched database
        ins["warm"] = [f"{WARM}/pythia8.ok", f"{WARM}/smodels_db.ok"]
    return ins


rule sample:
    output:
        points = f"{META}/{{scan}}/points.npz"
    params:
        sdir = lambda w: SCANS[w.scan]["dir"],
        seed = lambda w: SCANS[w.scan]["seed"],
        cfg = CFG_OPT
    shell:
        "{RULE_SCRIPTS}/r3mg_batch.py sample {params.cfg} --scan_dir {params.sdir} "
        "--seed {params.seed} --points_out {output.points}"


rule step:
    input:
        unpack(_step_inputs)
    output:
        manifest = f"{META}/{{scan}}/step_{{step}}.json"
    wildcard_constraints:
        step="|".join(re.escape(s) for s in STEP_NAMES)
    params:
        sdir = lambda w: SCANS[w.scan]["dir"],
        cfg = CFG_OPT,
        prev = lambda w, input: f"--prev_manifest {input.prev}" if getattr(input, "prev", "") else ""
    resources:
        # SModelS loads the whole results database (~1.4 GB pickle) into memory for
        # every model. Two batches doing that at once exceeded the CI container's
        # memory cap and the kernel killed one of them — a bare non-zero exit with
        # no traceback. Claiming a token that the local profile limits to 1 keeps
        # the heavy step serialised while every other step still runs in parallel.
        # Irrelevant on the cluster backends, where each batch is its own job.
        smodels_db = lambda w: 1 if w.step == "SModelS" else 0
    shell:
        "{RULE_SCRIPTS}/r3mg_batch.py step --step {wildcards.step} {params.cfg} "
        "--scan_dir {params.sdir} --points {input.points} {params.prev} "
        "--manifest_out {output.manifest}"


# The ntuple filename is derived from the scan-dir basename (naming contract
# with extractModels and the CI reference check): scan.<C>.<P> ->
# ntuple.<C>.<P>.root, anything else -> ntuple.0.0.root. The two modes have
# different (constant vs wildcard) shapes, hence two rule variants.
if MODE == "single":

    _sname = next(iter(SCANS))

    rule ntuple:
        input:
            last = f"{META}/{_sname}/step_{LAST_STEP}.json",
            points = f"{META}/{_sname}/points.npz"
        output:
            f"{SCANS[_sname]['dir']}/{_ntuple_basename(_sname)}"
        params:
            sdir = SCANS[_sname]["dir"],
            cfg = CFG_OPT
        shell:
            "{RULE_SCRIPTS}/r3mg_batch.py ntuple {params.cfg} --scan_dir {params.sdir} --points {input.points}"

else:

    rule ntuple:
        input:
            last = f"{META}/scan.{RUNID}.{{b}}/step_{LAST_STEP}.json",
            points = f"{META}/scan.{RUNID}.{{b}}/points.npz"
        output:
            f"{BATCH_PREFIX}.{{b}}/ntuple.{RUNID}.{{b}}.root"
        params:
            sdir = lambda w: f"{BATCH_PREFIX}.{w.b}",
            cfg = CFG_OPT
        shell:
            "{RULE_SCRIPTS}/r3mg_batch.py ntuple {params.cfg} --scan_dir {params.sdir} --points {input.points}"
