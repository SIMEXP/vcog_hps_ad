import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import roc_curve, auc


def plot_roc(y_test,y_score):

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    #plt.show()

def sbp_roc(sbp_dat):
    plt.figure(figsize=(6,6))
    #plt.subplot(2,2,1,colspan=2)
    plt.subplot2grid((3,2), (0,0), colspan=2,rowspan=2)
    plot_roc(sbp_dat[:,0],sbp_dat[:,1])
    plt.subplot2grid((3,2), (2,0))
    plot_roc(sbp_dat[sbp_dat[:,2]<=0,0],sbp_dat[sbp_dat[:,2]<=0,1])
    plt.subplot2grid((3,2), (2,1))
    plot_roc(sbp_dat[sbp_dat[:,2]>0,0],sbp_dat[sbp_dat[:,2]>0,1])
    plt.tight_layout()

def get_hm(clf,x,y):
    return np.array(clf.predict(x) == y,dtype=int)

def get_hm_labelonly(clf,x,y,label=1):
    # special HM calculation base only on a sub-group (denoted by label)
    hm_ = (y == label) & (clf.predict(x) == y)
    return np.array(hm_,dtype=int)
    
def get_hm_score_(lr,x,hm_y):

    w_coef = lr.coef_[0]

    df_data = lr.decision_function(x)
    print hm_y.shape
    print df_data.shape
    return np.array([hm_y[df_data<0], hm_y[df_data>0]])

def idx_decision(lr,data):
    return lr.decision_function(data)

def estimate_hitmiss(clf,x,y):
   
    label=1
    hm_results = []
    predictions =[]
    for i in range(len(y)):
        train_idx = np.array(np.hstack((np.arange(0,i),np.arange(i+1,len(y)))),dtype=int)
        #print train_idx.shape
        clf.fit(x[train_idx,:],y[train_idx])
        #print clf.predict(x[i,:]) == y[i]
        hm_results.append(int(clf.predict(x[i,:]) == y[i]))
        predictions.append(clf.predict(x[i,:]))
        #hm_results.append(int((y[i] == label) & (clf.predict(x[i,:]) == y[i]) ))#   clf.predict(x[i,:]) == y[i]))

    predictions = np.array(predictions)
    hm_results = np.array(hm_results)
    #print hm_results.shape
    clf.fit(x,y)
    return hm_results, predictions[:,0]
   
def show_hm_(lr,xtrain,hm_y,show_fig=True):

    w_coef = lr.coef_[0]

    idx_coef = np.argsort(w_coef)
    #print w_coef[idx_coef]

    lr_curve = [1./(1.+np.exp(-i)) for i in np.arange(-3.,3,0.1)]

    df_data = lr.decision_function(xtrain)
    
    if show_fig==True:
        plt.figure()
        y_tmp = lr.predict(xtrain)[hm_y==1]
        plt.plot(df_data[hm_y==1],y_tmp+np.random.normal(0, 0.1, size=len(y_tmp)),'ro')
        y_tmp = lr.predict(xtrain)[hm_y<=0]
        plt.plot(df_data[hm_y<=0],y_tmp+np.random.normal(0, 0.1, size=len(y_tmp)),'bo')

        # acc of the left side
        print (np.mean(hm_y[df_data<0]))
        # acc of the right side
        print (np.mean(hm_y[df_data>0]))
        print('Size',len(hm_y[df_data<0]),len(hm_y[df_data>0]))

        plt.plot(np.arange(-3.,3,0.1),lr_curve)
        plt.xlabel('Decision function\n ACC Left:' + "{0:.2f}".format(100*np.mean(hm_y[df_data<0]))+'%' + ' ACC Right:' + "{0:.2f}".format(100*np.mean(hm_y[df_data>0]))+'%')
        plt.ylabel('Prediction of the logistic regression')

        #plt.legend(['SZ','CTRL'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        #plt.legend(['Hit','Miss','logistic regression curve'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.legend(['Hit','Miss','logistic regression curve'],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
        plt.axis([-3, 3, -0.2, 1.5])
    return np.array([np.mean(hm_y[df_data<0]), np.mean(hm_y[df_data>0])])

def show_explicit_hm_(df_data,hm_y,labels=['Hit','Miss','logistic regression curve']):
    #sns.despine()
    sns.set_style("white")
    flatui = [ "#e74c3c","#3498db","#34495e", "#95a5a6", "#9b59b6", "#2ecc71"]
    sns.set_palette(flatui)
    
    lateral = df_data>0

    lr_curve = [1./(1.+np.exp(-i)) for i in np.arange(-3.,3,0.1)]
    
    randst = np.random.RandomState(seed=0)
    
    fig = plt.figure(dpi=250)
    y_tmp = lateral[hm_y==1]
    plt.plot(df_data[hm_y==1],y_tmp+randst.normal(0, 0.1, size=len(y_tmp)),'o')
    y_tmp = lateral[hm_y<=0]
    plt.plot(df_data[hm_y<=0],y_tmp+randst.normal(0, 0.1, size=len(y_tmp)),'o')


    # acc of the left side
    print (np.mean(hm_y[df_data<0]))
    # acc of the right side
    print (np.mean(hm_y[df_data>0]))
    print('Size',len(hm_y[df_data<0]),len(hm_y[df_data>0]))

    plt.plot(np.arange(-3.,3,0.1),lr_curve)
    plt.xlabel('Decision function ACC Left:' + "{0:.2f}".format(100*np.mean(hm_y[df_data<0]))+'%' + ' ACC Right:' + "{0:.2f}".format(100*np.mean(hm_y[df_data>0]))+'%')
    plt.ylabel('Prediction of the logistic regression')

    #plt.legend(['Hit','Miss','logistic regression curve'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.legend(labels,bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, mode="expand", borderaxespad=0.)
    plt.axis([-3, 3, -0.2, 1.5])
    return np.array([np.mean(hm_y[df_data<0]), np.mean(hm_y[df_data>0])])



def score(net_data_low_main,y,confounds,svm_model,lr_model,cfrm_model,st_model):
    # remove confounds
    net_data_low_tmp = cfrm_model.transform(confounds,net_data_low_main.reshape((net_data_low_main.shape[0],net_data_low_main.shape[1]*net_data_low_main.shape[2])))
    net_data_low = net_data_low_tmp.reshape((net_data_low_tmp.shape[0],net_data_low_main.shape[1],net_data_low_main.shape[2]))
    
    # subtypes extraction
    x = st_model.transform(net_data_low)
    x_svm = net_data_low.reshape(net_data_low.shape[0],net_data_low.shape[1]*net_data_low.shape[2])
    
    #svm_model.score(x_svm,y)
    df_data = lr_model.decision_function(x)
    lateralization = lr_model.predict(x)
    hm_y = get_hm(svm_model,x_svm,y)
    
    show_explicit_hm_(lateralization,df_data,hm_y)
    
def score_w(net_data_low_main,y,confounds,lr1_model,lr2_model,cfrm_model,st_model,idx_sz):
    net_data_low_tmp = cfrm_model.transform(confounds,net_data_low_main.reshape((net_data_low_main.shape[0],net_data_low_main.shape[1]*net_data_low_main.shape[2])))
    net_data_low = net_data_low_tmp.reshape((net_data_low_tmp.shape[0],net_data_low_main.shape[1],net_data_low_main.shape[2]))
    
    # subtypes extraction
    x = st_model.transform(net_data_low)[:,idx_sz]
    #x_svm = net_data_low.reshape(net_data_low.shape[0],net_data_low.shape[1]*net_data_low.shape[2])
    
    #svm_model.score(x_svm,y)
    df_data = lr2_model.decision_function(x)
    lateralization = lr2_model.predict(x)
    hm_y = get_hm(lr1_model,x,y)
    
    show_explicit_hm_(df_data,hm_y)
    
from sklearn import metrics
def results_row(y_ref,y_pred,lr_decision,pos_label=1,average='weighted'):
    row=[]
    row.append(metrics.accuracy_score(y_ref,y_pred))
    row.append(metrics.precision_recall_fscore_support(y_ref,y_pred,pos_label=pos_label,average=average)[:3])
    row.append(metrics.accuracy_score(y_ref[lr_decision<0],y_pred[lr_decision<0]))
    row.append(metrics.precision_recall_fscore_support(y_ref[lr_decision<0], y_pred[lr_decision<0],pos_label=pos_label,average=average)[:3])
    row.append(metrics.accuracy_score(y_ref[lr_decision>0],y_pred[lr_decision>0]))
    row.append(metrics.precision_recall_fscore_support(y_ref[lr_decision>0], y_pred[lr_decision>0],pos_label=pos_label,average=average)[:3])
    return np.hstack(row)

def classif_repo(y_ref,y_pred,lr_decision):

    print '##################################################################'
    print 'Main'
    print classification_report(y_ref, y_pred)
    print 'ACC: '+str(accuracy_score(y_ref,y_pred))
    print 'Right:'
    print classification_report(y_ref[lr_decision>0], y_pred[lr_decision>0])
    print 'ACC: '+str(accuracy_score(y_ref[lr_decision>0],y_pred[lr_decision>0]))
    print 'Left:'
    print classification_report(y_ref[lr_decision<0], y_pred[lr_decision<0])
    print 'ACC: '+str(accuracy_score(y_ref[lr_decision<0],y_pred[lr_decision<0]))
    print '##################################################################'
    
def classif_repo_bimode(y_ref,y_pred,lr_decision_sz,lr_decision_ctrl):

    print '##################################################################'
    print 'Main'
    print classification_report(y_ref, y_pred)
    print 'ACC: '+str(accuracy_score(y_ref,y_pred))
    print '#####################'
    print 'Right SZ:'
    print classification_report(y_ref[y_pred==1][lr_decision_sz[y_pred==1]>0], y_pred[y_pred==1][lr_decision_sz[y_pred==1]>0])
    print 'ACC: '+str(accuracy_score(y_ref[y_pred==1][lr_decision_sz[y_pred==1]>0],y_pred[y_pred==1][lr_decision_sz[y_pred==1]>0]))
    print 'Left SZ:'
    print classification_report(y_ref[y_pred==1][lr_decision_sz[y_pred==1]<=0], y_pred[y_pred==1][lr_decision_sz[y_pred==1]<=0])
    print 'ACC: '+str(accuracy_score(y_ref[y_pred==1][lr_decision_sz[y_pred==1]<=0],y_pred[y_pred==1][lr_decision_sz[y_pred==1]<=0]))
    print '#####################'
    print 'Right CTRL:'
    print classification_report(y_ref[y_pred==0][lr_decision_ctrl[y_pred==0]>0], y_pred[y_pred==0][lr_decision_ctrl[y_pred==0]>0])
    print 'ACC: '+str(accuracy_score(y_ref[y_pred==0][lr_decision_ctrl[y_pred==0]>0],y_pred[y_pred==0][lr_decision_ctrl[y_pred==0]>0]))
    print 'Left CTRL:'
    print classification_report(y_ref[y_pred==0][lr_decision_ctrl[y_pred==0]<=0], y_pred[y_pred==0][lr_decision_ctrl[y_pred==0]<=0])
    print 'ACC: '+str(accuracy_score(y_ref[y_pred==0][lr_decision_ctrl[y_pred==0]<=0],y_pred[y_pred==0][lr_decision_ctrl[y_pred==0]<=0]))
    print '##################################################################'
