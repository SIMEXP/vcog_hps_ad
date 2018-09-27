FROM jupyter/scipy-notebook

RUN pip install --no-cache-dir notebook==5.*

USER jovyan

RUN wget https://github.com/HanadS/vcog_hps_ad/blob/master/vcog_hpc_prediction_simulated_data.ipynb/archive/0.5.zip
RUN unzip 0.5.zip
RUN rm 0.5.zip

CMD ["jupyter", "notebook", "--ip", "0.0.0.0"]
