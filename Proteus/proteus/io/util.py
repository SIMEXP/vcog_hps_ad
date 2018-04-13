__author__ = 'Christian Dansereau'

# interact with a csv file

import pandas as pd
import numpy as np
import scipy.io
from os import listdir
from proteus.matrix import tseries as ts
from proteus.predic import clustering as clust
import nibabel as nib
import h5py


def write(dict_data, file_name, compress=4):
    with h5py.File(file_name, 'w') as hf:
        for ii in range(len(dict_data.keys())):
            item_id = dict_data.keys()[ii]
            hf.create_dataset(item_id, data=dict_data[item_id], compression="gzip", compression_opts=compress)


def load(file_name):
    with h5py.File(file_name, 'r') as hf:
        dict_data = {}
        print("List of arrays in this file: ", hf.keys())
        for ii in range(len(hf.keys())):
            item_id = hf.keys()[ii]
            dict_data[item_id] = np.array(hf.get(item_id))

    return dict_data


def organize_data(data, demograph):
    data_tmp = data.loc[demograph.index.values]
    data_tmp = data_tmp.iloc[data_tmp.iloc[:, 0].notnull().values]
    return data_tmp, demograph.loc[data_tmp.index.values]


def conv_conn4pred(source_path, destination_path):
    root_path = source_path
    list_files = listdir(root_path)
    print('Start loading ' + str(len(list_files)) + ' subjects...')
    Zscales = []
    Rscales = []
    index = []
    scale_list = []
    for i in range(len(list_files)):
        tmp_mat = scipy.io.loadmat(root_path + list_files[i])
        print(tmp_mat['subj_id'][0])
        if tmp_mat['nframes'][0] > 40:
            Z = tmp_mat['Z'][0]
            R = tmp_mat['R'][0]
            index.append(tmp_mat['subj_id'][0])
            scale_list = tmp_mat['scale_list'][0].astype(int)
            if len(Zscales) == 0:
                for j in range(len(Z)):
                    Zscales.append(Z[j].T)
                    Rscales.append(R[j].T)
            else:
                for j in range(len(Z)):
                    Zscales[j] = np.vstack((Zscales[j], Z[j].T))
                    Rscales[j] = np.vstack((Rscales[j], R[j].T))

    for j in range(len(Zscales)):
        df = pd.DataFrame(Zscales[j], index=index)
        df.to_csv(destination_path + 'model_Z_conn_scale' + str(scale_list[j]) + '.csv')
        df = pd.DataFrame(Rscales[j], index=index)
        df.to_csv(destination_path + 'model_R_conn_scale' + str(scale_list[j]) + '.csv')

    return scale_list


def grabStability(root_path, part_, nclusters=12, windowsize=20):
    list_files = listdir(root_path)
    data_array = []
    subj_list = []
    k = 0
    for i in range(len(list_files)):
        try:
            tmp_mat = scipy.io.loadmat(root_path + list_files[i])
            print tmp_mat.keys()
            if tmp_mat['vol'].shape[3] > windowsize + 20:
                subj_list.append(tmp_mat['subj_id'][0])
                ts_ = ts.get_ts(tmp_mat['vol'], part_.get_data())
                tmp_data2 = clust.getWindowCluster(ts_, nclusters, windowsize).mean(axis=0)

                if k == 0:
                    data_array = tmp_data2[np.newaxis, :]
                else:
                    data_array = np.vstack((data_array, tmp_data2[np.newaxis, :]))
                k += 1
        except:
            print('Exception: ' + root_path + list_files[i])
    return pd.DataFrame(data_array, index=subj_list)


def grabConnectivityWindowsStats(root_path, part_, windowsize=20):
    list_files = listdir(root_path)
    means_array = []
    stds_array = []
    subj_list = []
    k = 0
    for i in range(len(list_files)):
        # try:
        if list_files[i].split('.')[-1] == 'mat':
            tmp_mat = scipy.io.loadmat(root_path + list_files[i])
            tmp_subjid = tmp_mat['subj_id'][0]
            tmp_vol = tmp_mat['vol']
        else:
            tmp_vol = nib.load(root_path + list_files[i]).get_data()
            tmp_vol = np.swapaxes(np.swapaxes(tmp_vol, 0, 3), 1, 2)
            tmp_subjid = list_files[i].split('_')[1]

        print(tmp_subjid)

        if tmp_vol.shape[3] > 40:
            subj_list.append(tmp_subjid)
            ts_ = ts.get_ts(tmp_vol, part_.get_data())
            windows_val = clust.getWindows(ts_, windowsize)
            print windows_val.shape
            tmp_data_mean = windows_val.mean(axis=0)
            tmp_data_std = windows_val.std(axis=0)
            if k == 0:
                means_array = tmp_data_mean[np.newaxis, :]
                stds_array = tmp_data_std[np.newaxis, :]
            else:
                means_array = np.vstack((means_array, tmp_data_mean[np.newaxis, :]))
                stds_array = np.vstack((stds_array, tmp_data_std[np.newaxis, :]))
            k += 1
            # except:
            #    print('Exception: ' + root_path + list_files[i])
    return pd.DataFrame(means_array, index=subj_list), pd.DataFrame(stds_array, index=subj_list)
