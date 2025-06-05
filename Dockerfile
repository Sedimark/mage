FROM mageai/mageai:latest

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
    pyenv install 3.11.9; \
    pyenv global 3.11.9

ENV PATH="/root/.pyenv/shims:/root/.pyenv/bin:$PATH"

COPY default_repo/ /home/src/default_repo/

RUN python3.11 -m pip install /home/src/default_repo/utils/*.whl 

RUN rm /home/src/default_repo/utils/*.whl
