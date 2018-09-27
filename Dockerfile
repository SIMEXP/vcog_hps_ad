FROM jupyter/scipy-notebook

RUN pip install --no-cache-dir notebook==5.*

USER jovyan

ENTRYPOINT ["jupyter", "notebook", "--ip=*"]
