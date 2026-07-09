FROM chdonner/msml

RUN apt-get update && apt-get install -y \
    gcc gfortran \
    libeccodes0 libeccodes-dev

RUN pip install --upgrade pip

COPY . /workspace/

RUN pip install --no-cache-dir -e /workspace/

# Uncomment for GPU support:
# RUN pip install --upgrade "jax[cuda12]"

RUN useradd -m appuser && chsh -s /bin/bash appuser
RUN chown -R appuser:appuser /workspace
RUN echo 'PS1="\u@\h:\w$ "' >> /home/appuser/.bashrc
USER appuser
WORKDIR /workspace