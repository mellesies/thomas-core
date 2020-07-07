# The Dockerfile tells Docker how to construct the image.
FROM mellesies/thomas-base-python3:latest

LABEL maintainer="Melle Sieswerda <m.sieswerda@iknl.nl>"

ARG NB_USER=jupyter
ARG NB_UID=1001
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Copy package
COPY . /usr/local/python/thomas-core/

# Make sure the contents of our repo are also in ${HOME}
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
RUN chown -R ${NB_UID} /usr/local/python/thomas-core/
USER ${NB_USER}

WORKDIR /usr/local/python/
RUN pip install ./thomas-core

# JupyterLab runs on port 8888
EXPOSE 8888

# CMD /bin/bash
WORKDIR /usr/local/python/thomas-core
# WORKDIR ${HOME}/notebooks
CMD jupyter lab --ip=0.0.0.0 --allow-root --LabApp.token=''
