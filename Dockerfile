# The Dockerfile tells Docker how to construct the image.
FROM python:3

LABEL maintainer="Melle Sieswerda <m.sieswerda@iknl.nl>"

# ------------------------------------------------------------------------------
# Preliminaries
# ------------------------------------------------------------------------------
# Create a default user
ARG NB_USER=jupyter
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}
ENV THOMAS_DIR /home/${NB_USER}/thomas

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Make sure ~/.local/bin exists and is in PATH
RUN mkdir -p ${HOME}/.local/bin
ENV PATH="${PATH}:${HOME}/.local/bin"

# Set the root password
USER root
RUN echo "root:root" | chpasswd

# Install nodejs >= 12
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt-get update
RUN apt-get install -y nodejs

# Install yarn properly. The version of yarn that's available by default is
# just weird.
RUN apt-get remove cmdtest
RUN npm install --global yarn

# ------------------------------------------------------------------------------
# JupyterLab settings (specifically: keyboard shortcuts & port)
# ------------------------------------------------------------------------------
COPY lab ${HOME}/.jupyter/lab
EXPOSE 8888

# ------------------------------------------------------------------------------
# Python package installation
# ------------------------------------------------------------------------------
COPY notebooks ${THOMAS_DIR}/thomas-core/notebooks
COPY tests ${THOMAS_DIR}/thomas-core/tests
COPY thomas ${THOMAS_DIR}/thomas-core/thomas
COPY setup.py ${THOMAS_DIR}/thomas-core
COPY utest.py ${THOMAS_DIR}/thomas-core
COPY test.sh ${THOMAS_DIR}/thomas-core
COPY README.md ${THOMAS_DIR}/thomas-core

# Make sure files are owned by
RUN chown -R ${NB_UID}:${USER} ${HOME}
RUN chown -R ${NB_UID}:${USER} ${THOMAS_DIR}/thomas-core/

# ------------------------------------------------------------------------------
# Run as ${USER} !
# ------------------------------------------------------------------------------
USER ${USER}
ENV PATH="${PATH}:${USER}/.local/bin"

WORKDIR ${THOMAS_DIR}/
RUN pip install ./thomas-core[jupyter-dev,client]

WORKDIR ${THOMAS_DIR}/thomas-core
RUN ./utest.py

# WORKDIR ${HOME}/notebooks
RUN jupyter serverextension enable jupyterlab
CMD jupyter lab --ip=0.0.0.0 --allow-root --LabApp.token=''
