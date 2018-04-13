__author__ = 'Christian Dansereau'

import numpy as np
from sklearn.cluster import KMeans
from proteus.predic import clustering as cls
from proteus.matrix import tseries as ts
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.model_selection import ShuffleSplit
from proteus.predic import prediction


def st_multi_fit(confounds, x, nSubtypes=3):
    st_crm = []
    for ii in range(len(x)):
        crm = prediction.ConfoundsRm(confounds, x[ii])
        st = clusteringST()
        st.fit_network(crm.transform(confounds, x[ii]), nSubtypes=nSubtypes)
        st_crm.append([crm, st])
    return st_crm


def st_multi_transform(st_crm, confounds, x, nsplit=1):
    W = []
    for ii in range(len(st_crm)):
        W.append(st_crm[ii][1].compute_weights(st_crm[ii][0].transform(confounds, x[ii]),
                                               mask_part=_chunks(x[ii][0, :], nsplit)))
    w_ = np.hstack(W)
    w_ = reshapeW(w_)
    return w_


def _chunks(l, n):
    w = len(l) / (n)
    k = 1
    l = np.zeros_like(l)
    for i in xrange(0, len(l), w):
        if k == (n):
            l[i:] = k
            break
        else:
            l[i:i + w] = k
        k += 1
    return l


class clusteringST:
    '''
    Identification of sub-types for prediction
    '''

    def __init__(self, verbose=True):
        self.verbose = verbose

    def fit(self, net_data_low, nSubtypes=3, reshape_w=True):
        # net_data_low = net_data_low_main.copy()
        self.flag_2level = False
        self.nnet_cluster = net_data_low.shape[1]
        self.nSubtypes = nSubtypes

        # ind_low_scale = cls.get_ind_high2low(low_res_template,orig_template)
        # self.ind_low_scale = ind_low_scale

        # net_data_low --> Dimensions: nSubjects, nNetwork_low, nNetwork
        # net_data_low = transform_low_scale(ts_data,self.ind_low_scale)
        # self.net_data_low = net_data_low

        self.normalized_net_template = []
        for i in range(net_data_low.shape[1]):
            # average template
            if nSubtypes < 1:
                self.normalized_net_template.append(np.zeros_like(net_data_low[0, i, :]).astype(float))
            else:
                self.normalized_net_template.append(np.mean(net_data_low[:, i, :], axis=0))
                # self.normalized_net_template.append(np.zeros_like(net_data_low[0,i,:])).astype(float))

            # indentity matrix of the corelation between subjects
            # tmp_subj_identity = np.corrcoef(net_data_low[:,i,:])
            # ind_st = cls.hclustering(tmp_subj_identity,nSubtypes)
            # subjects X network_nodes
            ind_st = cls.hclustering(net_data_low[:, i, :], nSubtypes)

            for j in range(nSubtypes):
                if j == 0:
                    st_templates_tmp = np.median(net_data_low[:, i, :][ind_st == j + 1, :], axis=0)[np.newaxis, ...]
                else:
                    st_templates_tmp = np.vstack((st_templates_tmp,
                                                  np.median(net_data_low[:, i, :][ind_st == j + 1, :], axis=0)[
                                                      np.newaxis, ...]))

            # st_templates --> Dimensions: nNetwork_low, nSubtypes, nNetwork
            if i == 0:
                self.st_templates = st_templates_tmp[np.newaxis, ...]
            else:
                self.st_templates = np.vstack((self.st_templates, st_templates_tmp[np.newaxis, ...]))
            del st_templates_tmp

        # calculate the weights for each subjects
        self.W = self.compute_weights(net_data_low, self.st_templates)


        if reshape_w:
            return reshapeW(self.W)
        else:
            return self.W

    def _norm_subjects(self, data, ref=[]):
        if len(data.shape) == 2:

            ref_avg_rmaps = ref.mean()
            avrg_rmaps = data.mean(1)
            scaling_factor = ref_avg_rmaps / avrg_rmaps
            return data * scaling_factor.reshape(-1, 1)
        else:
            ref_avg_rmaps = np.array(self.normalized_net_template).mean(1)
            avrg_rmaps = data.mean(2)
            scaling_factor = ref_avg_rmaps / avrg_rmaps
            print(ref_avg_rmaps.shape, avrg_rmaps.shape, scaling_factor.shape)
            return np.swapaxes(np.swapaxes(data, 0, 2) * np.swapaxes(scaling_factor, 0, 1), 0, 2)

    def _robust_st(self, net_data_low, nSubtypes, n_iter=50):
        bs_cluster = []
        n = net_data_low.shape[0]
        stab_ = np.zeros((n, n)).astype(float)
        rs = ShuffleSplit(net_data_low.shape[0], n_iter=n_iter, test_size=.05, random_state=1)
        for train, test in rs:
            # indentity matrix of the corelation between subjects
            ind_st = cls.hclustering(net_data_low[train, :], nSubtypes)
            mat_ = (cls.ind2matrix(ind_st) > 0).astype(float)
            for ii in range(len(train)):
                stab_[train, train[ii]] += mat_[:, ii]

        stab_ = stab_ / n_iter
        ms = KMeans(nSubtypes)
        ind = ms.fit_predict(stab_)
        # row_clusters = linkage(stab_, method='ward')
        # ind = fcluster(row_clusters, nSubtypes, criterion='maxclust')
        return ind + 1, stab_

    def fit_robust(self, net_data_low, nSubtypes=3, reshape_w=True, stab_thereshold=0.5):
        self.flag_2level = False
        self.nnet_cluster = net_data_low.shape[1]
        self.nSubtypes = nSubtypes

        self.normalized_net_template = []
        for i in range(net_data_low.shape[1]):
            # average template
            self.normalized_net_template.append(np.mean(net_data_low[:, i, :], axis=0))
            # self.normalized_net_template.append(np.zeros_like(net_data_low[0,i,:]))

            # indentity matrix of the corelation between subjects
            # ind_st = cls.hclustering(net_data_low[:,i,:],nSubtypes)
            ind_st, stab_ = self._robust_st(net_data_low[:, i, :] - self.normalized_net_template[-1], nSubtypes)

            for j in range(nSubtypes):
                mask_stable = (stab_[ind_st == j + 1, :].mean(0) > stab_thereshold)[ind_st == j + 1]
                if self.verbose: print('Robust: new N ', mask_stable.sum(), ' old N ', mask_stable.shape)
                data_ = net_data_low[ind_st == j + 1, i, :][mask_stable, :]
                if j == 0:
                    st_templates_tmp = np.median(data_, axis=0)[np.newaxis, ...]
                else:
                    st_templates_tmp = np.vstack((st_templates_tmp, np.median(data_, axis=0)[np.newaxis, ...]))

            # st_templates --> Dimensions: nNetwork_low, nSubtypes, nNetwork
            if i == 0:
                self.st_templates = st_templates_tmp[np.newaxis, ...]
            else:
                self.st_templates = np.vstack((self.st_templates, st_templates_tmp[np.newaxis, ...]))
            del st_templates_tmp

        # calculate the weights for each subjects
        self.W = self.compute_weights(net_data_low, self.st_templates)
        if reshape_w:
            return reshapeW(self.W)
        else:
            return self.W

    def fit_robust_network(self, net_data_low, nSubtypes=3, reshape_w=True, stab_thereshold=0.5):
        self.flag_2level = False
        self.nnet_cluster = 1
        self.nSubtypes = nSubtypes
        # net_data_low --> Dimensions: nSubjects, nNetwork_low, nNetwork

        self.normalized_net_template = []
        # average template
        self.normalized_net_template.append(np.mean(net_data_low[:, :], axis=0))
        # self.normalized_net_template.append(np.zeros_like(net_data_low[0,:]))
        # indentity matrix of the corelation between subjects
        ind_st, stab_ = self._robust_st(net_data_low - self.normalized_net_template[-1], nSubtypes)

        for j in range(nSubtypes):
            mask_stable = (stab_[ind_st == j + 1, :].mean(0) > stab_thereshold)[ind_st == j + 1]
            if self.verbose: print('Robust: new N ', mask_stable.sum(), ' old N ', mask_stable.shape)
            data_ = net_data_low[ind_st == j + 1, :][mask_stable, :]
            if j == 0:
                st_templates_tmp = np.median(data_, axis=0)[np.newaxis, ...]
            else:
                st_templates_tmp = np.vstack((st_templates_tmp, np.median(data_, axis=0)[np.newaxis, ...]))

        # st_templates --> Dimensions: nNetwork_low,nSubtypes, nNetwork
        self.st_templates = st_templates_tmp[np.newaxis, ...]
        del st_templates_tmp
        # calculate the weights for each subjects
        self.W = self.compute_weights(net_data_low, self.st_templates)
        if reshape_w:
            return reshapeW(self.W)
        else:
            return self.W

    def fit_network(self, net_data_low, nSubtypes=3, reshape_w=True):
        self.flag_2level = False
        self.nnet_cluster = 1
        self.nSubtypes = nSubtypes
        # self.scalers = []
        # net_data_low --> Dimensions: nSubjects, nNetwork_low, nNetwork

        self.normalized_net_template = []
        # average template
        # self.scalers.append(preprocessing.StandardScaler())
        # net_data_low = self.scalers[-1].fit_transform(net_data_low_)
        self.normalized_net_template.append(np.mean(net_data_low, axis=0))
        # self.normalized_net_template.append(np.zeros_like(net_data_low[0,:]))
        # identity matrix of the correlation between subjects
        ind_st = cls.hclustering(net_data_low, nSubtypes)

        for j in range(nSubtypes):
            data_tmp = np.median(net_data_low[ind_st == j + 1, :], axis=0)[np.newaxis, ...]
            if j == 0:
                st_templates_tmp = data_tmp
            else:
                st_templates_tmp = np.vstack((st_templates_tmp, data_tmp))

        # st_templates --> Dimensions: nNetwork_low,nSubtypes, nNetwork
        self.st_templates = st_templates_tmp[np.newaxis, ...]
        del st_templates_tmp
        # calculate the weights for each subjects
        self.W = self.compute_weights(net_data_low, self.st_templates)

        if reshape_w:
            return reshapeW(self.W)
        else:
            return self.W

    def _fit_2level(self, net_data_low_l1, net_data_low_l2, nSubtypes_l1=5, nSubtypes_l2=2, reshape_w=True):

        # Discontinued function

        self.flag_2level = True
        self.nnet_cluster = net_data_low_l1.shape[1]
        self.nSubtypes = nSubtypes_l1 * nSubtypes_l2
        self.nSubtypes_l1 = nSubtypes_l1
        self.nSubtypes_l2 = nSubtypes_l2

        # net_data_low --> Dimensions: nSubjects, nNetwork_low, nNetwork
        self.net_data_low = net_data_low_l1
        self.net_data_low_l2 = net_data_low_l2

        ####
        # LEVEL 1
        ####
        # st_templates --> Dimensions: nNetwork_low, nSubtypes, nNetwork
        st_templates = []
        for i in range(net_data_low_l1.shape[1]):
            # indentity matrix of the corelation between subjects
            ind_st = cls.hclustering(net_data_low_l1[:, i, :], nSubtypes_l1)

            for j in range(nSubtypes_l1):
                if j == 0:
                    st_templates_tmp = net_data_low_l1[:, i, :][ind_st == j + 1, :].mean(axis=0)[np.newaxis, ...]
                else:
                    st_templates_tmp = np.vstack(
                        (st_templates_tmp, net_data_low_l1[:, i, :][ind_st == j + 1, :].mean(axis=0)[np.newaxis, ...]))

            if i == 0:
                st_templates = st_templates_tmp[np.newaxis, ...]
            else:
                st_templates = np.vstack((st_templates, st_templates_tmp[np.newaxis, ...]))

        self.st_templates_l1 = st_templates

        # calculate the weights for each subjects
        # W --> Dimensions: nSubjects,nNetwork_low, nSubtypes
        net_data_low_l2_tmp = np.vstack((net_data_low_l1, net_data_low_l2))
        self.W_l1 = self.compute_weights(net_data_low_l2_tmp, self.st_templates_l1)

        ####
        # LEVEL 2                                                                                                                                                                                            
        ####
        # st_templates --> Dimensions: nNetwork_low, nSubtypes, nNetwork
        st_templates = []
        # st_templates = self.st_templates_l1.copy()
        # st_templates = st_templates[:,:,np.newaxis,:]
        for i in range(net_data_low_l2.shape[1]):

            # Iterate on all the Level1 subtypes (normal variability subtypes)
            for k in range(self.st_templates_l1.shape[1]):
                # Find the L1 subtype
                max_w = np.max(self.W_l1[:, i, :], axis=1)
                mask_selected_subj = (self.W_l1[:, i, k] == max_w)
                template2substract = self.st_templates_l1[i, k, :]
                if np.sum(mask_selected_subj) <= 3:
                    print('Less then 2 subjects for network: ' + str(i) + ' level1 ST: ' + str(k))
                    for j in range(nSubtypes_l2):
                        if (k == 0) & (j == 0):
                            st_templates_tmp = self.st_templates_l1[i, k, :][np.newaxis, ...]
                        else:
                            st_templates_tmp = np.vstack(
                                (st_templates_tmp, self.st_templates_l1[i, k, :][np.newaxis, ...]))

                else:
                    # indentity matrix of the corelation between subjects
                    ind_st = cls.hclustering(net_data_low_l2_tmp[:, i, :][mask_selected_subj, ...] - template2substract,
                                             nSubtypes_l2)
                    # ind_st = cls.hclustering(net_data_low[:,i,:],nSubtypes)
                    if len(np.unique(ind_st)) < nSubtypes_l2:
                        print('Clustering generated less class then asked nsubjects: ' + str(
                            len(ind_st)) + ' network: ' + str(i) + ' level1 ST: ' + str(k))
                        # if (i==6) & (k==3):
                        # print ind_st
                    for j in range(nSubtypes_l2):
                        if (k == 0) & (j == 0):
                            st_templates_tmp = (net_data_low_l2_tmp[:, i, :][mask_selected_subj, ...][ind_st == j + 1,
                                                :] - template2substract).mean(axis=0)[np.newaxis, ...]
                        else:
                            st_templates_tmp = np.vstack((st_templates_tmp, (
                                net_data_low_l2_tmp[:, i, :][mask_selected_subj, ...][ind_st == j + 1,
                                :] - template2substract).mean(axis=0)[np.newaxis, ...]))

            if i == 0:
                st_templates = st_templates_tmp[np.newaxis, ...]
            else:
                print(st_templates.shape, st_templates_tmp.shape)
                st_templates = np.vstack((st_templates, st_templates_tmp[np.newaxis, ...]))

        self.st_templates_l2 = st_templates

        # calculate the weights for each subjects
        self.W_l2 = self.compute_weights(net_data_low_l2, self.st_templates_l2)
        if reshape_w:
            return reshapeW(self.W_l2)
        else:
            return self.W_l2

    def compute_weights(self, net_data_low, st_templates=[], mask_part=[]):

        if st_templates == []:
            st_templates = self.st_templates
        # calculate the weights for each subjects
        # W = np.zeros((net_data_low.shape[0],st_templates.shape[0],st_templates.shape[1]))
        for j in range(st_templates.shape[0]):
            average_template = self.normalized_net_template[j]
            if len(net_data_low.shape) == 2:
                # net_data_low = self.scalers[j].transform(net_data_low_)
                rmaps = net_data_low - average_template
            else:
                # net_data_low[:,j,:] = self.scalers[j].transform(net_data_low_[:,j,:])
                rmaps = net_data_low[:, j, :] - average_template
            st_rmap = st_templates[j, :, :] - average_template
            tmp_rmap = self._compute_w(rmaps, st_rmap, mask_part)
            if j == 0:
                W = np.zeros((net_data_low.shape[0], st_templates.shape[0], tmp_rmap.shape[1]))
            W[:, j, :] = np.nan_to_num(tmp_rmap)

            return W

    def _compute_w_global(self, X, ref):
        range_ = 1
        if len(X.shape) == 3:
            # multiple networks
            for net in range(X.shape[1]):
                if len(ref.shape) > 2:
                    range_ = ref.shape[1]
                w_global = np.corrcoef(ref[net, ...], X[:, net, :])[range_:, 0:range_]
        else:
            # One network
            if len(ref.shape) > 1:
                range_ = ref.shape[0]
            w_global = np.corrcoef(ref, X)[range_:, 0:range_]

        return w_global

    def _compute_w(self, X, ref, mask_part=[]):
        if mask_part != []:
            # sub_w based on partition
            w_ = []
            list_id = np.unique(mask_part)
            for idx in np.delete(list_id, np.where(list_id == 0)):
                mask_ = mask_part == idx
                w_.append(self._compute_w_global(X[..., mask_], ref[..., mask_]))
            w_ = np.hstack(w_)
        else:
            # global mode, no sub-partition
            w_ = self._compute_w_global(X, ref)
        return w_

    def _compute_weights_l2(self, net_data_low):
        corrected_ndl = net_data_low.copy()
        W_l1 = self.compute_weights(net_data_low, self.st_templates_l1)

        # calculate the weights for each subjects
        for i in range(net_data_low.shape[1]):
            for k in range(self.st_templates_l1.shape[1]):
                # Find the L1 subtype
                max_w = np.max(W_l1[:, i, :], axis=1)
                mask_selected_subj = (W_l1[:, i, k] == max_w)
                corrected_ndl[mask_selected_subj, i, :] = corrected_ndl[mask_selected_subj, i,
                                                          :] - self.st_templates_l1[i, k, :]

        return self.compute_weights(corrected_ndl, self.st_templates_l2)

    def reshapeW_inv(self, W):

        # reshape the matrix from [subjects, Nsubtypes, weights] to [subjects, vector of weights]
        xw = W.reshape((W.shape[0], self.nnet_cluster, W.shape[1]/(self.nnet_cluster)))
        return xw

    def transform(self, net_data_low, mask_part=[], reshape_w=True):
        '''
            Calculate the weights for each sub-types previously computed
        '''
        # compute the low scale version of the data
        # net_data_low = transform_low_scale(ts_data,self.ind_low_scale)

        #if self.flag_2level:
        #    # calculate the weights for each subjects
            # W = self.compute_weights(net_data_low,self.st_templates_l2)
        #    W = self._compute_weights_l2(net_data_low)
        #else:
            # calculate the weights for each subjects
        W = self.compute_weights(net_data_low, self.st_templates, mask_part)

        if reshape_w:
            return reshapeW(W)
        else:
            return W


    def fit_dev(self, net_data, nnet_cluster='auto', nSubtypes=3):
        self.nnet_cluster = nnet_cluster
        self.nSubtypes = nSubtypes

        if nnet_cluster == 'auto':
            # self.nnet_cluster = self.getClusters(net_data)
            self.valid_cluster, self.valid_net_idx = self.get_match_network(net_data, nnet_cluster, algo='meanshift')
        else:
            self.valid_cluster, self.valid_net_idx = self.get_match_network(net_data, nnet_cluster, algo='kmeans')

        # self.valid_cluster = self.clust_list
        # self.valid_net_idx = range(len(self.valid_cluster))
        for i in range(net_data.shape[0]):
            if i == 0:
                self.assign_net = self.assigneDist(net_data[i, :, :], self.valid_cluster, self.valid_net_idx)
            else:
                self.assign_net = np.vstack(
                    ((self.assign_net, self.assigneDist(net_data[i, :, :], self.valid_cluster, self.valid_net_idx))))
        print('Size of the new data map: ', self.assign_net.shape)
        # group subjects with the most network classifing them together
        # compute the consensus clustering
        self.consensus = cls.hclustering(self.assign_net, self.nSubtypes)
        # save the centroids in a method
        self.clf_subtypes = NearestCentroid()
        self.clf_subtypes.fit(self.assign_net, self.consensus)
        self.consensus = self.clf_subtypes.predict(self.assign_net)
        # print "score: ", self.clf_subtypes.score(self.assign_net,self.consensus)

        return self.consensus


def transform_low_scale(ts_data, ind_low_scale, normalize=True):
    # compute the connectivity for at template at a given resolution
    allsubj_lowxhigh_conn = []
    for i in range(len(ts_data)):
        ind_data = ts_data[i]
        tmp_ts_array = []
        max_id_scale = ind_low_scale.max()
        # compute the time series for the low scale
        for j in range(max_id_scale):
            if j == 0:
                tmp_ts_array = np.mean(ind_data[ind_low_scale == j + 1, :], axis=0)
            else:
                tmp_ts_array = np.vstack((tmp_ts_array, np.mean(ind_data[ind_low_scale == j + 1, :], axis=0)))

        # calculation of the correlation between each timeseries
        allsubj_lowxhigh_conn.append(
            np.corrcoef(np.vstack((ind_data, tmp_ts_array)))[-max_id_scale:, :][:, :-max_id_scale])
    # reorder the dimensions
    allsubj_lowxhigh_conn = np.dstack(allsubj_lowxhigh_conn)
    net_data_low = np.swapaxes(np.swapaxes(allsubj_lowxhigh_conn, 1, 2), 0, 1)
    # net_data_low --> Dimensions: nSubjects, nNetwork_low, nNetwork

    # replace nan values by zero (corrcoef output nan whe series are constant!)
    net_data_low = np.nan_to_num(net_data_low)
    return net_data_low


def reshape_netwise(data_scale):
    # Reshape with the following dim: nSubjects, nfeatures, nfeatures
    for i in range(0, data_scale.shape[0]):
        if i == 0:
            all_subjmat = ts.vec2mat(data_scale[i, :])[np.newaxis, ...]
        else:
            all_subjmat = np.vstack((all_subjmat, ts.vec2mat(data_scale[i, :])[np.newaxis, ...]))

    # print all_subjmat.shape
    return all_subjmat


def format_nets(data, select_idx=[]):
    list_data = []
    for n in range(0, len(data)):
        # tranform in matrix format
        if len(select_idx) > 0:
            clean_data = data[n][select_idx, :]
        else:
            clean_data = data[n]
        tmp_mat = reshape_netwise(clean_data)
        for i in range(0, tmp_mat.shape[2]):
            select_x = tmp_mat[:, :, i]
            list_data.append(select_x)
    return list_data


def convSubScale(net_data, indtoconv):
    new_data = np.zeros((net_data.shape[0], net_data.shape[1], np.max(indtoconv)))
    for i in range(np.max(indtoconv)):
        new_data[:, :, i] = np.mean(net_data[:, :, indtoconv == i + 1], axis=2)
    return new_data


def reshapeW(W):
    # reshape the matrix from [subjects, Nsubtypes, weights] to [subjects, vector of weights]
    xw = W.reshape((W.shape[0], W.shape[1] * W.shape[2]))
    return xw
