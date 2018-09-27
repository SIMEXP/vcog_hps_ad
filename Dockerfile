FROM jupyter/scipy-notebook

RUN pip install --no-cache-dir notebook==5.*

USER jovyan

CMD ["jupyter", "notebook", "--ip", "0.0.0.0"]
