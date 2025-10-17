# Dockerfile - builds an amd64 Miniconda image
ARG TARGETPLATFORM
FROM --platform=${TARGETPLATFORM:-linux/amd64} continuumio/miniconda3:latest

# set working dir
WORKDIR /app

# copy environment file and install environment
COPY environment.yml /tmp/environment.yml
RUN conda install -n base -c conda-forge mamba && \
    mamba env create -f /tmp/environment.yml && \
    conda clean -afy && \
    rm /tmp/environment.yml

# install PyCO2SYS beta
RUN pip install --no-deps git+https://github.com/mvdh7/PyCO2SYS@v2.0.0-b5

# ensure conda environment is active for interactive shells and subsequent RUNs
SHELL ["bash", "-c"]
RUN echo "conda activate project2" >> ~/.bashrc 
ENV PATH=/opt/conda/envs/project2/bin:$PATH

# copy project code
COPY . /app

# default command for debugging
ENTRYPOINT ["bash"]
