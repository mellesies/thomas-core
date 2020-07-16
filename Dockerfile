# The Dockerfile tells Docker how to construct the image.
FROM mellesies/thomas-base-python3:latest

LABEL maintainer="Melle Sieswerda <m.sieswerda@iknl.nl>"

ARG NB_USER=jupyter
ARG NB_UID=1001
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

USER root

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Install widget first, so the (editable) install of 'core' overwrites
# the PyPI version of thomas-core.
RUN pip install thomas-jupyter-widget
RUN jupyter labextension install --minimize=False thomas-jupyter-widget

# Copy & install package
COPY . /usr/local/python/thomas-core/
WORKDIR /usr/local/python/
RUN pip install ./thomas-core


RUN chown -R ${NB_UID} ${HOME}
RUN chown -R ${NB_UID} /usr/local/python/thomas-core/
USER ${NB_USER}

# JupyterLab runs on port 8888
EXPOSE 8888

# CMD /bin/bash
WORKDIR /usr/local/python/thomas-core
# WORKDIR ${HOME}/notebooks
CMD jupyter lab --ip=0.0.0.0 --allow-root --LabApp.token=''
