# Dockerfile
FROM chdonner/msml

RUN apt-get update && apt-get install -y \
    gcc gfortran \
    libeccodes0 libeccodes-dev

# Install Black, Ruff, Mypy, Pre-commit
RUN pip install --upgrade pip 
# && \
#    pip install black ruff mypy pre-commit

# Copy all files from the current directory to /workspace
COPY . /workspace/

# Optional: copy your requirements

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt 
# && pre-commit install

## Uncomment the following line to install JAX with CUDA 12 support.
# RUN pip install --upgrade "jax[cuda12]"

RUN mypy --install-types --non-interactive

# Create a non-root user with a home directory and set shell to bash
RUN useradd -m appuser
RUN chsh -s /bin/bash appuser

# Set ownership of /workspace
RUN chown -R appuser:appuser /workspace

# Configure shell prompt
RUN mkdir -p /home/appuser
RUN echo 'PS1="\u@\h:\w $"' >> /home/appuser/.bashrc

# Switch to the non-root user
USER appuser

# Set workdir
WORKDIR /workspace