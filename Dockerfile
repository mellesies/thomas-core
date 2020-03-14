# The Dockerfile tells Docker how to construct the image.
FROM mellesies/thomas-base-python3:latest

LABEL maintainer="Melle Sieswerda <m.sieswerda@iknl.nl>"

# Copy package
COPY . /usr/local/python/thomas-core/

WORKDIR /usr/local/python/
RUN pip install ./thomas-core

# JupyterLab runs on port 8888
EXPOSE 8888

# CMD /bin/bash
WORKDIR /usr/local/python/thomas-core/notebooks
CMD jupyter lab --ip 0.0.0.0 --allow-root --LabApp.token=''
