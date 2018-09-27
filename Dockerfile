FROM jupyter/scipy-notebookRUN 

RUN pip install --no-cache-dir notebook==5.*

USER jovyan

ENTRYPOINT ["jupyter", "notebook", "--ip=*"]
