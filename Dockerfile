# Specify the image from which you are working
FROM gitlab-registry.cern.ch/linuxsupport/alma9-base:latest

# Put the current repo (the one in which this Dockerfile resides) in the directory specified here
# Note that this directory is created on the fly and does not need to reside in the repo already
COPY . /workdir

# Go into the directory specified here (again, it will create the directory if it doesn't already exist)
WORKDIR /workdir/

# Set default shell to bash
SHELL ["/bin/bash", "-c"]

# Install Run3ModelGen. PYTHIA8_CACHE_DIR makes `pixi run build` save the compiled
# pythia8 to a stable path baked into the image; CI docker jobs (which run in a fresh
# checkout with a fresh pixi env) restore from it instead of recompiling pythia8 per
# job (see .setup_docker in .gitlab-ci.yml and scripts/setup_pythia8.sh).
ENV PYTHIA8_CACHE_DIR=/opt/pythia8-cache
# SMODELS_CACHEDIR makes `pixi run build` prefetch SModelS's results-database pickle
# to a stable path baked into the image; CI docker jobs point here (see .setup_docker)
# to reuse it instead of re-downloading the (hundreds-of-MB) database per job.
ENV SMODELS_CACHEDIR=/opt/smodels-cache
RUN curl -fsSL https://pixi.sh/install.sh | bash && \
    . ~/.bash_profile && \
    pwd && \
    ls  && \
    mkdir run && \
    pixi list && \
    pixi run python --version && \
    pixi run which gcc && \
    pixi run which cc && \
    pixi run which c++ && \
    pixi run build && \
    ls

# Append pixi path and setup script to bashrc
RUN echo "export PATH=$PATH:/root/.pixi/bin" >> ~/.bashrc && \
    echo "source /workdir/build/setup.sh" >> ~/.bashrc

# Define commands to be run when entering container
CMD [ "cd run && pixi shell" ]

ENTRYPOINT [ "/bin/bash", "-i", "-c" ]