import math

# Point-sampled priors (flat/fixed/grid/importance): sample once per batch, then one
# rule instance per (tool, batch), gated through the per-step manifests. Per-model
# files inside the scan dir are NOT declared outputs (they legitimately don't exist
# for failed models); the manifests in WORK are the declared, atomic rule products.
#
# WORK, not META: a manifest describes the contents of its scan dir, and the step /
# ntuple bodies abort if it names files the scan dir no longer holds. The two must
# therefore live or die together — see the Snakefile docstring.
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
# ── per-rule time budget ────────────────────────────────────────────────────────
# A group job's time limit is the SUM of its rules' runtimes, so one flat `runtime`
# for every step spends eight times the slow tool's budget: seven of the eight steps
# finish in minutes while each claims SModelS' whole allowance. With runtime=120 that
# summed to 16h for ~1.5h of work and capped a batch at ~2100 models -- below the 10k
# per-directory cap, so time was the binding constraint rather than the cap.
#
# Budget per tool instead, scaled from each tool's measured per-model cost, so nearly
# the whole allowance lands on SModelS (~87% of the real work). The scaling reads
# num_models, so the request tracks the batch size automatically. At the 10k cap this
# declares ~17h against the 24h partition limit, and a batch really takes ~11h.
#
# Seconds per ATTEMPTED model, measured on the 200-model reference batch in
# run/smodels_random (SModelS with LO cross sections and combineSRs off). Tools
# downstream of SPheno see only the ~58% of models yielding a usable spectrum; that
# fraction is folded in, so every figure below is per attempted model.
_SECONDS_PER_MODEL = {
    "sample":     0.005,
    "prep_input": 0.005,
    "SPheno":     0.135,
    "softsusy":   0.020,
    "micromegas": 0.115,   # 0.198 measured x 0.58 usable
    "superiso":   0.095,   # 0.164 x 0.58
    "gm2calc":    0.010,   # 0.017 x 0.58
    "feynhiggs":  0.145,   # 0.250 x 0.58
    "SModelS":    3.420,   # 5.900 x 0.58
    "ntuple":     0.040,
}
# Headroom on the mean. The per-model cost varies a lot point to point (SModelS time
# scales with how many analyses match a spectrum), and the means above come from ~110
# usable models, so the mean itself is uncertain. A wall-clock kill loses the batch's
# whole ~11h of work, so buy 50% on every rule.
_TIME_MARGIN = 1.50
_RUNTIME_FLOOR = 10        # minutes; never request less

# ...and give the slow tool everything the others leave under the partition cap.
# Unrequested allowance is NOT available at runtime: the summed group request becomes
# the job's --time and slurm kills at it, so time left unclaimed is forfeited rather
# than held in reserve. SModelS is also the only rule that can realistically overrun
# (the rest need 1-25 min), so raising ITS budget alone converts the remainder into
# tolerance without inflating trivial rules and hurting backfill. At the 10k cap that
# takes SModelS from 1.5x to ~2.2x its expected cost for free.
_PARTITION_MINUTES = 24 * 60   # 'general' MaxTime=1-00:00:00
_GROUP_RESERVE = 20            # keep the summed request clear of the cap
_SLOW_STEP = "SModelS"
_SLOW_MARGIN_MAX = 2.50        # ceiling, so a small batch does not ask for 21h


def _scaled(name, n):
    per = _SECONDS_PER_MODEL[name]
    return max(_RUNTIME_FLOOR, math.ceil(_TIME_MARGIN * per * n / 60))


def _rule_runtime(name):
    '''Minutes to request for one rule over this config's num_models.'''
    if name not in _SECONDS_PER_MODEL:
        return 120         # unknown tool: keep the historical flat budget
    n = max(1, int(CFG.get("num_models") or 1))
    base = _scaled(name, n)
    if name != _SLOW_STEP:
        return base
    others = sum(_scaled(k, n) for k in _SECONDS_PER_MODEL if k != _SLOW_STEP)
    remainder = _PARTITION_MINUTES - _GROUP_RESERVE - others
    capped = math.ceil(_SLOW_MARGIN_MAX * _SECONDS_PER_MODEL[name] * n / 60)
    return max(base, min(remainder, capped))


_PRODUCER_OF = {s["output_dir"]: s["name"] for s in STEPS if s.get("output_dir")}
_STEP_BY_NAME = {s["name"]: s for s in STEPS}


def _gate_of(step_name):
    '''Name of the step whose output `step_name` consumes (None if it has no input).'''
    indir = _STEP_BY_NAME[step_name].get("input_dir")
    return _PRODUCER_OF.get(indir) if indir else None


def _step_inputs(wildcards):
    ins = {"points": f"{WORK}/{wildcards.scan}/points.npz"}
    gate = _gate_of(wildcards.step)
    if gate:
        ins["prev"] = f"{WORK}/{wildcards.scan}/step_{gate}.json"
    if wildcards.step == "SModelS":
        # in-process smodels needs the built pythia8 + prefetched database
        ins["warm"] = [f"{WARM}/pythia8.ok", f"{WARM}/smodels_db.ok"]
    return ins


rule sample:
    output:
        points = f"{WORK}/{{scan}}/points.npz"
    params:
        sdir = lambda w: SCANS[w.scan]["dir"],
        seed = lambda w: SCANS[w.scan]["seed"],
        cfg = CFG_OPT
    resources:
        runtime = _rule_runtime("sample")
    shell:
        "{RULE_SCRIPTS}/r3mg_batch.py sample {params.cfg} --scan_dir {params.sdir} "
        "--seed {params.seed} --points_out {output.points}"


rule step:
    input:
        unpack(_step_inputs)
    output:
        manifest = f"{WORK}/{{scan}}/step_{{step}}.json"
    wildcard_constraints:
        step="|".join(re.escape(s) for s in STEP_NAMES)
    params:
        sdir = lambda w: SCANS[w.scan]["dir"],
        cfg = CFG_OPT,
        prev = lambda w, input: f"--prev_manifest {input.prev}" if getattr(input, "prev", "") else ""
    resources:
        # Per-TOOL time budget (see _rule_runtime): SModelS gets ~83% of the group's
        # allowance because it is ~87% of the work. A flat per-step runtime made the
        # group's summed limit 8x the slow tool's need.
        runtime = lambda w: _rule_runtime(w.step),
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
            steps = expand(f"{WORK}/{_sname}/step_{{step}}.json", step=STEP_NAMES),
            points = f"{WORK}/{_sname}/points.npz"
        output:
            f"{SCANS[_sname]['dir']}/{_ntuple_basename(_sname)}"
        params:
            sdir = SCANS[_sname]["dir"],
            cfg = CFG_OPT
        resources:
            runtime = _rule_runtime("ntuple")
        shell:
            "{RULE_SCRIPTS}/r3mg_batch.py ntuple {params.cfg} --scan_dir {params.sdir} --points {input.points}"

else:

    rule ntuple:
        input:
            # every step's manifest — see the single-mode rule above
            steps = expand(f"{WORK}/scan.{RUNID}.{{{{b}}}}/step_{{step}}.json",
                           step=STEP_NAMES),
            points = f"{WORK}/scan.{RUNID}.{{b}}/points.npz"
        output:
            f"{BATCH_PREFIX}.{{b}}/ntuple.{RUNID}.{{b}}.root"
        params:
            sdir = lambda w: f"{BATCH_PREFIX}.{w.b}",
            cfg = CFG_OPT
        resources:
            runtime = _rule_runtime("ntuple")
        shell:
            "{RULE_SCRIPTS}/r3mg_batch.py ntuple {params.cfg} --scan_dir {params.sdir} --points {input.points}"
