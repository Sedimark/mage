FROM mageai/mageai:latest AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget \
    llvm libncursesw5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/bin:$PATH"
RUN curl -fsSL https://pyenv.run | bash

SHELL ["/bin/bash", "-c"]

RUN eval "$(pyenv init --path)"; \
    eval "$(pyenv init -)"; \
    pyenv install 3.11;
    
FROM mageai/mageai:latest

ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

COPY --from=builder /root/.pyenv /root/.pyenv

COPY default_repo/ /home/src/default_repo/

RUN pyenv local 3.11 && pyenv exec python3.11 -m pip install default_repo/libs/fleviden-0.4.0-py3-none-any.whl && pyenv local system && pyenv global system

RUN python --version
