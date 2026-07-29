# mode=extract — batch extraction over a published EOS run directory
# ($EOSPATH/<ClusterId>/<ProcId>/scan...tar.gz + ntuple...root), one rule
# instance per ProcId subdir (driven by extractModels.py --eos_dir).
# Each instance calls the unchanged extractModels.py CLI (ModelExtractor),
# which writes to $EOSPATH/SelectedModels/<ClusterId>/Models/<subdir>/<ProcId>.

if MODE == "extract":

    EOS_DIR = str(config["eos_dir"]).rstrip("/")
    CLUSTER = os.path.basename(EOS_DIR)
    # The selection is an awkward-array expression full of quotes. Cluster
    # executors re-invoke snakemake with the whole --config line embedded in a
    # scheduler submit description, and HTCondor's parser rejects the quotes it
    # then contains ("Found illegal unescaped double-quote"), so the
    # entrypoints ship it base64-encoded — the same trick snakemake uses for
    # its own --default-resources. Plain selection= stays supported for direct
    # (local) invocations, which never go through a submit file.
    if config.get("selection_b64"):
        SELECTION = base64.b64decode(str(config["selection_b64"])).decode()
    else:
        SELECTION = str(config.get("selection", "akarr['SS_m_h']!=-1"))
    PROCIDS = [p.strip() for p in str(config["procids"]).split(",") if p.strip() != ""]
    WORK = str(config.get("work_dir", ".")).rstrip("/")
    MARK = os.path.normpath(f"{WORK}/r3mg_extract")

    rule extract_batch:
        output:
            touch(f"{MARK}/{CLUSTER}.{{p}}.ok")
        wildcard_constraints:
            p=r"\d+"
        params:
            tar = lambda w: f"{EOS_DIR}/{w.p}/scan.{CLUSTER}.{w.p}.tar.gz",
            ntuple = lambda w: f"{EOS_DIR}/{w.p}/ntuple.{CLUSTER}.{w.p}.root",
            sel = SELECTION
        shell:
            "extractModels.py --scan_dir {params.tar} --root_file {params.ntuple} "
            "--selection {params.sel:q}"

    rule extract_all:
        input:
            expand(f"{MARK}/{CLUSTER}.{{p}}.ok", p=PROCIDS)
