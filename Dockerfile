FROM jupyter/scipy-notebook:137a295ff71b

RUN pip install --no-cache-dir notebook==5.*

ENV NB_USER jovyan
ENV NB_UID 1000
ENV HOME /home/${NB_USER}

COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}


CMD ["jupyter", "notebook", "--ip", "0.0.0.0"]
