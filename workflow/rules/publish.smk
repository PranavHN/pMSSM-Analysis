# Batch-mode packaging + EOS publish, byte-mirroring the historical condor
# worker flow: tar the scan dir, then copy tarball + config yaml + ntuple to
# $EOSPATH/<runid>/<b>/. The tarball stage folds the
# xsec-augmented SModelS_xsec/<n>.slha copies back over the SPheno/ member
# paths (and drops the SModelS_xsec dir), so published tarballs are
# byte-layout-identical to the legacy in-place-mutation era — pMSSMFactory
# sees no difference.

if MODE == "batch":

    rule tarball:
        input:
            ntuple = f"{BATCH_PREFIX}.{{b}}/ntuple.{RUNID}.{{b}}.root"
        output:
            f"{BATCH_PREFIX}.{{b}}.tar.gz"
        params:
            sdir = lambda w: f"{BATCH_PREFIX}.{w.b}",
            cfg = CFG_OPT
        shell:
            "{RULE_SCRIPTS}/r3mg_batch.py tarball {params.cfg} --scan_dir {params.sdir} --tar_out {output}"

    # Unless keep_local is set, publishing is a MOVE, not a copy: the rule body
    # verifies every published file (presence + byte size) and then removes the
    # batch's scan dir and tarball, so a published run costs nothing outside
    # $EOSPATH. Only r3mg/ bookkeeping stays behind, which is what keeps the
    # run resumable — see _final_targets in the Snakefile for why the ntuples
    # must drop out of the target list when this is on.
    rule publish:
        input:
            tar = f"{BATCH_PREFIX}.{{b}}.tar.gz",
            ntuple = f"{BATCH_PREFIX}.{{b}}/ntuple.{RUNID}.{{b}}.root"
        output:
            # NOT touch(): the rule body writes this itself, so the marker lands
            # BEFORE the local copies are deleted. With touch() the marker was
            # created after the body exited, leaving a window where a killed job
            # had published to EOS, deleted the local scan dir, and recorded
            # nothing — a batch resume could not rebuild (its inputs were gone)
            # and could not skip (no marker).
            marker = f"{META}/scan.{RUNID}.{{b}}/published.ok"
        params:
            sdir = lambda w: f"{BATCH_PREFIX}.{w.b}",
            cleanup = "" if KEEP_LOCAL else "--cleanup"
        shell:
            "{RULE_SCRIPTS}/r3mg_batch.py publish --scan_dir {params.sdir} --tarball {input.tar} "
            f"--runid {RUNID} " + "--procid {wildcards.b} --marker {output.marker} {params.cleanup}"
