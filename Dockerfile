ARG PYTHON_VERSION=3.7-alpine
FROM python:${PYTHON_VERSION}
LABEL maintainer="Suraj Iyer <suraj.iyer@vodafoneziggo.com>"
# ARG PYTHON_LIBS_PATH
# RUN test -n "${PYTHON_LIBS_PATH}" || (echo "PYTHON_LIBS_PATH not set" && false)

# Set proxy server, replace host:port with values for your servers
# ENV http_proxy host:port
# ENV https_proxy host:port

# Install linux packages
RUN apk update \
    && apk add --no-cache alpine-sdk \
    && apk add --no-cache --virtual build-dependencies \
        curl \
        htop \
        unzip \
        unrar \
        tree \
        freetds-dev \
        bash

# Create basic project directory
RUN mkdir ~/project
WORKDIR ~/project

# Copy extra python libraries
RUN mkdir ~/pythonlibs
COPY . ~/pythonlibs
ENV PYTHONPATH ~/pythonlibs/python-data-utils/python

# Install Python packages
RUN pip install -r ~/pythonlibs/python-data-utils/requirements.txt

# Jupyter lab extensions
RUN apk add --no-cache nodejs \
    && jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    && jupyter labextension install @jupyterlab/toc \
    && jupyter labextension install @jupyterlab/git \
    && pip install --upgrade jupyterlab-git \
    && jupyter serverextension enable --py jupyterlab_git \
    && jupyter labextension install @mflevine/jupyterlab_html \
    && jupyter labextension install jupyterlab-spreadsheet \
    && pip install ipympl \
    && jupyter labextension install jupyter-matplotlib

# Create start-jupyter.sh
RUN "cd ~/project; jupyter lab --no-browser --LabApp.token='' --ip=127.0.0.1 --port=8888" > ~/start-jupyter.sh
EXPOSE 8888

CMD ["/bin/sh"]