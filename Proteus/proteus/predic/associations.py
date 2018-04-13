import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
from nistats import glm as nsglm
from sklearn import metrics


def association_glm(var1, var2):
    # we assume that the var1 is a numerical variable already encoded


    # Encode variables
    x = var2

    # Sanity check
    if (x == 0).sum() != x.shape[0]:

        # Normalize
        x = (x - x.mean()) / x.std()
        y = (var1 - var1.mean()) / var1.std()
        # print y.shape,x.shape
        # print x
        # GLM
        contrast = [0, 1]
        x_ = np.vstack((np.ones_like(x), x)).T
        labels, regression_result = nsglm.session_glm(y[:, np.newaxis], x_)
        cont_results = nsglm.compute_contrast(labels, regression_result, contrast, contrast_type='t')
        pval = cont_results.p_value()[0]
        return cont_results.stat()[0], cont_results.p_value()[0][0][0]


    else:
        print '### Error nothing to regress ###'
        return np.NAN, np.NAN


def plot_st_w(stw, contrast):
    n_plot = stw.shape[1]

    n_x = np.ceil(np.sqrt(n_plot))
    n_y = np.ceil(n_plot / (n_x * 1.))
    plt.figure(figsize=(1.5 * n_x, 2 * n_y))
    f = plt.gcf()

    f.subplots_adjust(hspace=0.5)
    f.subplots_adjust(wspace=0.5)
    for i in range(n_plot):

        plt.subplot(1 * n_y, n_x, i + 1)
        ax = plt.gca()
        # plt.scatter(contrast,stw[:,i],color='k', alpha=0.5)
        data = [stw[:, i][contrast == j] for j in np.unique(contrast)]

        # Association test
        pval = association_glm(stw[:, i], contrast)[1]
        # print data
        # sns.violinplot(data)
        violin_parts = plt.violinplot(data, np.unique(contrast), points=15, widths=0.5,
                                      showmeans=False, showextrema=True, showmedians=True)

        for pc in violin_parts['bodies']:
            pc.set_color('black')
            pc.set_facecolor('black')
            pc.set_edgecolor('black')
        violin_parts['cbars'].set_color('black')
        violin_parts['cmedians'].set_color('black')
        violin_parts['cmaxes'].set_color('black')
        violin_parts['cmins'].set_color('black')
        # violin_parts['cmeans'].set_color('black')

        plt.xticks(np.unique(contrast), ['HC', 'Patho'])
        plt.ylim([-1, 1])
        plt.yticks(np.arange(-1, 1.1, 0.5))
        # ax.yaxis.grid(True,'major')
        ax.xaxis.grid(False)

        # ax.set_yticks([-0.5, 0.5, 0.5], minor=True)
        if pval < 0.05:
            ax.set_title('* ' + str(i + 1))
        else:
            ax.set_title(str(i + 1))
        if (i % n_x) > 0:
            ax.minorticks_on
            ax.get_yaxis().set_visible(False)
            ax.get_yaxis().set_ticks([-0.5, 0, 0.5], minor=False)
            ax.yaxis.grid(True, which='major')

        else:
            plt.ylabel('Weights')
            ax.get_yaxis().set_visible(True)


