ARG NVCR_TAG=25.08
FROM nvcr.io/nvidia/jax:${NVCR_TAG}-py3

# Install direct dependencies, but not source code.
WORKDIR /stage
COPY pyproject.toml .
COPY src/olmax/__init__.py src/olmax/__init__.py
COPY src/olmax/version.py src/olmax/version.py
RUN pip install --no-cache-dir '.[all]' && \
    pip uninstall -y ai2-olmax && \
    rm -rf *
