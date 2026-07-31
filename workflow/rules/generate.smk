# Point-sampled priors (flat/fixed/grid/importance): sample once per batch, then one
# rule instance per (tool, batch), gated through the per-step manifests. Per-model
# files inside the scan dir are NOT declared outputs (they legitimately don't exist
# for failed models); the manifests in META are the declared, atomic rule products.
#
# Gating follows the DATA DEPENDENCY — each step waits on the step that produces its
# `input_dir` — not the position in the config list. Previously it was a linear chain
# in config order, which meant micromegas, superiso, gm2calc, feynhiggs and SModelS
# were serialised behind each other even though all five read the SAME SPheno output:
# a model that micromegas could say nothing about was then never attempted by
# superiso/gm2calc/feynhiggs either, for no reason beyond list order. Now they all
# gate on SPheno and are independent of one another (which also lets the DAG run them
# concurrently). NB this changes `attempted` per step, so the CI per-step reference
# counts must be regenerated.
_PRODUCER_OF = {s["output_dir"]: s["name"] for s in STEPS if s.get("output_dir")}
_STEP_BY_NAME = {s["name"]: s for s in STEPS}


def _gate_of(step_name):
    '''Name of the step whose output `step_name` consumes (None if it has no input).'''
    indir = _STEP_BY_NAME[step_name].get("input_dir")
    return _PRODUCER_OF.get(indir) if indir else None


def _step_inputs(wildcards):
    ins = {"points": f"{META}/{wildcards.scan}/points.npz"}
    gate = _gate_of(wildcards.step)
    if gate:
        ins["prev"] = f"{META}/{wildcards.scan}/step_{gate}.json"
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
            # EVERY step's manifest, not just the last one. Gating is a tree now (each
            # step waits on the step producing its input_dir), so depending on the last
            # step no longer transitively pulls in the others — softsusy, micromegas,
            # superiso and gm2calc feed nothing downstream and would drop out of the DAG
            # entirely, silently producing an ntuple missing their branches.
            steps = expand(f"{META}/{_sname}/step_{{step}}.json", step=STEP_NAMES),
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
            # every step's manifest — see the single-mode rule above
            steps = expand(f"{META}/scan.{RUNID}.{{{{b}}}}/step_{{step}}.json",
                           step=STEP_NAMES),
            points = f"{META}/scan.{RUNID}.{{b}}/points.npz"
        output:
            f"{BATCH_PREFIX}.{{b}}/ntuple.{RUNID}.{{b}}.root"
        params:
            sdir = lambda w: f"{BATCH_PREFIX}.{w.b}",
            cfg = CFG_OPT
        shell:
            "{RULE_SCRIPTS}/r3mg_batch.py ntuple {params.cfg} --scan_dir {params.sdir} --points {input.points}"
