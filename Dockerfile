FROM jupyter/scipy-notebookRUN 

pip install --no-cache-dir notebook==5.*

USER jovyan

ENTRYPOINT ["jupyter", "notebook", "--ip=*"]
