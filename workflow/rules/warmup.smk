# Shared-singleton warm-up, BEFORE any fan-out: SModelS's bundled pythia8 is
# compiled into the pixi env on first use and the results-database pickle is
# downloaded into SMODELS_CACHEDIR. Both are process-shared; building them once
# on the submit host stops N concurrent first-use builds (the flock inside
# setup_pythia8.sh remains the backstop). Flags live under build/ because the
# warmed state is a property of the environment, not of any one run.

localrules: warm_pythia8, warm_smodels_db

rule warm_pythia8:
    output:
        touch(f"{WARM}/pythia8.ok")
    shell:
        "bash {SCRIPTS}/setup_pythia8.sh"

rule warm_smodels_db:
    output:
        touch(f"{WARM}/smodels_db.ok")
    shell:
        "bash {SCRIPTS}/setup_smodels_db.sh"
