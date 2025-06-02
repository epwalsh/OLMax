FROM nvcr.io/nvidia/jax:25.04-py3

# Install direct dependencies, but not source code.
COPY pyproject.toml .
COPY src/olmax/__init__.py src/olmax/__init__.py
COPY src/olmax/version.py src/olmax/version.py
RUN pip install --no-cache-dir '.[all]' && \
    pip uninstall -y ai2-olmax && \
    rm -rf *
