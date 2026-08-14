FROM python:3.12-slim

# rdkit + opencv (via ultralytics/easyocr) need libGL and glib at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The package version comes from git tags via hatch-vcs, and .git is not in the
# build context (see .dockerignore). Without this the build fails outright, and
# a wrong version silently trips the weights compatibility gate in weights.py —
# cser-relmatcher requires >=0.3.0. CI passes the release tag.
ARG VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION

COPY pyproject.toml README.md ./
COPY structflo/ ./structflo/
COPY annotate/ ./annotate/
COPY webapp/ ./webapp/
RUN pip install --no-cache-dir .

# Every model this app uses is downloaded on first run: YOLO detector + LPS +
# relmatcher from HF Hub, DECIMER via pystow, EasyOCR's craft/english weights.
# That is several GB, so point them all at one directory and mount it — a bare
# container re-downloads the lot on every start.
ENV HF_HOME=/cache/huggingface \
    PYSTOW_HOME=/cache/pystow \
    EASYOCR_MODULE_PATH=/cache/easyocr \
    MPLCONFIGDIR=/cache/matplotlib
VOLUME /cache

EXPOSE 8000

ENTRYPOINT ["sf-web"]
# --preload pays the model-loading cost at startup instead of on the first
# upload. Drop it if you would rather the container come up immediately.
CMD ["--host", "0.0.0.0", "--port", "8000", "--preload"]
