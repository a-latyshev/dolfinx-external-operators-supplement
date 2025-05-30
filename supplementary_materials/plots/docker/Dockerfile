FROM ubuntu@sha256:dc17125eaac86538c57da886e494a34489122fb6a3ebb6411153d742594c2ddc
# 24.04, noble-20250415.1

RUN apt-get -qq update && \
    apt-get -yq --with-new-pkgs upgrade && \
    apt-get -y --fix-missing install python3-full texlive-latex-base texlive-fonts-extra && \
    rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/virtualenv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH=${VIRTUAL_ENV}/bin:$PATH

RUN python3 -m pip install --no-cache-dir numpy==2.2.6 matplotlib==3.10.3 mpltools==0.2.0 pandas==2.2.3
