FROM tensorflow/tensorflow:latest-py3

COPY requirements.txt /tmp/pip-tmp/
RUN pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends git zsh curl wget vim man \
    && git clone --recursive https://github.com/sorin-ionescu/prezto.git "${ZDOTDIR:-$HOME}/.zprezto" \
    && /bin/zsh -c 'setopt EXTENDED_GLOB; for rcfile in "${ZDOTDIR:-$HOME}"/.zprezto/runcoms/^README.md(.N); do ln -s "$rcfile" "${ZDOTDIR:-$HOME}/.${rcfile:t}"; done' \
    && chsh -s /bin/zsh \
    && curl https://raw.githubusercontent.com/shah/engineering-sandbox-debian/master/p10k.zsh > $HOME/.p10k.zsh \
    && curl https://raw.githubusercontent.com/shah/engineering-sandbox-debian/master/zpreztorc > $HOME/.zpreztorc
