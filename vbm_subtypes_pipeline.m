%% subtype on ADNI1 CN and AD subjects

clear all

path_data = '/home/atam/adni_vbm_octave/adni1/';
path_model = '/home/atam/adni_vbm_octave/adni1_bl_vbm_model_niak.csv';
path_mask = '/home/atam/yasser_templates/rc1mni_icbm152_t1_tal_nlin_asym_09c_6.nii'; 
path_out = '/home/atam/adni1_vbm_adcn_subtypes_20171209/';

scale = 7;

path_results = strcat(path_out, num2str(scale), 'clus');

%% set up files_in structure

files_in.mask = path_mask;
files_in.model = path_model;

%% filter out those with failed QC in model
[conf_model,list_subject,cat_names] = niak_read_csv(files_in.model);
qc_col = find(strcmp('vbm_qc',cat_names));
mask_qc = logical(conf_model(:,qc_col));
conf_model = conf_model(mask_qc,:);
list_subject = list_subject(mask_qc);

%% filter out the MCI patients
mci_col = find(strcmp('MCI',cat_names));
nan_mci = isnan(conf_model(:,mci_col));
% mask out the nan values
conf_model = conf_model(~nan_mci,:);
list_subject = list_subject(~nan_mci);
% find mci subjects and mask them out
mask_mci = logical(conf_model(:,mci_col));
conf_model = conf_model(~mask_mci,:);
list_subject = list_subject(~mask_mci);

%% grab oldest mwrc1rl_T1 from each subject's folder

folds = dir(path_data);
folds = {folds.name};
folds = folds(~ismember(folds,{'.','..','logs','qc','models'}));

for ss = 1:length(folds)
    % From folder name, grab subject ID
    sub_fold = folds{ss};
    tmp = strsplit(sub_fold,'_');
    rid = tmp{5};
    sid = strcat('s',rid);
    % Only grab the file if subject passed QC
    if any(ismember(list_subject,sid))
        files_in.data.network_1.(sid) = [path_data filesep sub_fold];
    end
end
%% options
opt.folder_out = path_results;
opt.scale = 1;
opt.stack.regress_conf = {'gender','age_scan','mean_gm','tiv'};

opt.subtype.nb_subtype = scale;

%% run the pipeline
%opt.psom.qsub_options = '--mem=8G  --account rrg-pbellec --time=00-03:00 --ntasks=1 --cpus-per-task=1  ';

opt.flag_test = false;
[pipe,opt] = niak_pipeline_subtype(files_in,opt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% extract weights in MCI subjects from ADNI1
clear all

path_data = '/home/atam/adni_vbm_octave/adni1/';
path_model = '/home/atam/adni_vbm_octave/adni1_bl_vbm_model_niak.csv';
path_mask = '/home/atam/yasser_templates/rc1mni_icbm152_t1_tal_nlin_asym_09c_6.nii';  
path_out = '/home/atam/adni1_vbm_adcn_subtypes_20171209/';

scale = 7;

path_results = strcat(path_out, num2str(scale), 'clus/', 'mci_weights');

%% set up files_in structure

files_in.mask = path_mask;
files_in.model = path_model;
files_in.subtype.network_1 = strcat('/home/atam/adni1_vbm_adcn_subtypes_20171209/',num2str(scale),'clus/networks/network_1/subtype_network_1.mat');

%% filter out those with failed QC in model
[conf_model,list_subject,cat_names] = niak_read_csv(files_in.model);
qc_col = find(strcmp('vbm_qc',cat_names));
mask_qc = logical(conf_model(:,qc_col));
conf_model = conf_model(mask_qc,:);
list_subject = list_subject(mask_qc);

%% filter just the MCI patients
mci_col = find(strcmp('MCI',cat_names));
nan_mci = isnan(conf_model(:,mci_col));
% mask out the nan values
conf_model = conf_model(~nan_mci,:);
list_subject = list_subject(~nan_mci);
% find mci subjects and mask everyone else out
mask_mci = logical(conf_model(:,mci_col));
conf_model = conf_model(mask_mci,:);
list_subject = list_subject(mask_mci);

%% grab oldest mwrc1rl_T1 from each subject's folder

folds = dir(path_data);
folds = {folds.name};
folds = folds(~ismember(folds,{'.','..','logs','qc','models'}));

for ss = 1:length(folds)
    % From folder name, grab subject ID
    sub_fold = folds{ss};
    tmp = strsplit(sub_fold,'_');
    rid = tmp{5};
    sid = strcat('s',rid);
    % Only grab the file if subject passed QC
    if any(ismember(list_subject,sid))
        files_in.data.network_1.(sid) = [path_data filesep sub_fold];
    end
end
%% options
opt.folder_out = path_results;
opt.scale = 1;
opt.stack.regress_conf = {'gender','age_scan','mean_gm','tiv'};

opt.subtype.nb_subtype = scale;

%% run the pipeline
%opt.psom.qsub_options = '--mem=8G  --account rrg-pbellec --time=00-03:00 --ntasks=1 --cpus-per-task=1  ';

opt.flag_test = false;
[pipe,opt] = niak_pipeline_subtype(files_in,opt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% extract weights in MCI subjects from ADNI2
clear all

path_data = '/home/atam/adni_vbm_octave/adni2/';
path_model = '/home/atam/adni_vbm_octave/adni2_bl_vbm_model_niak.csv';
path_mask = '/home/atam/yasser_templates/rc1mni_icbm152_t1_tal_nlin_asym_09c_6.nii';  
path_out = '/home/atam/adni1_vbm_adcn_subtypes_20171209/';

scale = 7;

path_results = strcat(path_out, num2str(scale), 'clus/', 'adni2_MCI_weights');

%% set up files_in structure

files_in.mask = path_mask;
files_in.model = path_model;
files_in.subtype.network_1 = strcat('/home/atam/adni1_vbm_adcn_subtypes_20171209/',num2str(scale),'clus/networks/network_1/subtype_network_1.mat');

%% filter out those with failed QC in model
[conf_model,list_subject,cat_names] = niak_read_csv(files_in.model);
qc_col = find(strcmp('vbm_qc',cat_names));
mask_qc = logical(conf_model(:,qc_col));
conf_model = conf_model(mask_qc,:);
list_subject = list_subject(mask_qc);

%% filter just the MCI patients
mci_col = find(strcmp('MCI',cat_names));
nan_mci = isnan(conf_model(:,mci_col));
% mask out the nan values
conf_model = conf_model(~nan_mci,:);
list_subject = list_subject(~nan_mci);
% find mci subjects and mask everyone else out
mask_mci = logical(conf_model(:,mci_col));
conf_model = conf_model(mask_mci,:);
list_subject = list_subject(mask_mci);

%% grab oldest mwrc1rl_T1 from each subject's folder

folds = dir(path_data);
folds = {folds.name};
folds = folds(~ismember(folds,{'.','..','logs','qc','models'}));

for ss = 1:length(folds)
    % From folder name, grab subject ID
    sub_fold = folds{ss};
    tmp = strsplit(sub_fold,'_');
    rid = tmp{4};
    sid = strcat('s',rid);
    % Only grab the file if subject passed QC
    if any(ismember(list_subject,sid))
        files_in.data.network_1.(sid) = [path_data filesep sub_fold];
    end
end
%% options
opt.folder_out = path_results;
opt.scale = 1;
opt.stack.regress_conf = {'gender','age_scan','mean_gm','tiv'};

opt.subtype.nb_subtype = scale;

%% run the pipeline
%opt.psom.qsub_options = '--mem=8G  --account rrg-pbellec --time=00-03:00 --ntasks=1 --cpus-per-task=1  ';

opt.flag_test = false;
[pipe,opt] = niak_pipeline_subtype(files_in,opt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% extract weights in AD subjects from ADNI2
clear all

path_data = '/home/atam/adni_vbm_octave/adni2/';
path_model = '/home/atam/adni_vbm_octave/adni2_bl_vbm_model_niak.csv';
path_mask = '/home/atam/yasser_templates/rc1mni_icbm152_t1_tal_nlin_asym_09c_6.nii';  
path_out = '/home/atam/adni1_vbm_adcn_subtypes_20171209/';

scale = 7;

path_results = strcat(path_out, num2str(scale), 'clus/', 'adni2_AD_weights');

%% set up files_in structure

files_in.mask = path_mask;
files_in.model = path_model;
files_in.subtype.network_1 = strcat('/home/atam/adni1_vbm_adcn_subtypes_20171209/',num2str(scale),'clus/networks/network_1/subtype_network_1.mat');

%% filter out those with failed QC in model
[conf_model,list_subject,cat_names] = niak_read_csv(files_in.model);
qc_col = find(strcmp('vbm_qc',cat_names));
mask_qc = logical(conf_model(:,qc_col));
conf_model = conf_model(mask_qc,:);
list_subject = list_subject(mask_qc);

%% filter just the AD patients
ad_col = find(strcmp('AD',cat_names));
nan_ad = isnan(conf_model(:,ad_col));
% mask out the nan values
conf_model = conf_model(~nan_ad,:);
list_subject = list_subject(~nan_ad);
% find mci subjects and mask everyone else out
mask_ad = logical(conf_model(:,ad_col));
conf_model = conf_model(mask_ad,:);
list_subject = list_subject(mask_ad);

%% grab oldest mwrc1rl_T1 from each subject's folder

folds = dir(path_data);
folds = {folds.name};
folds = folds(~ismember(folds,{'.','..','logs','qc','models'}));

for ss = 1:length(folds)
    % From folder name, grab subject ID
    sub_fold = folds{ss};
    tmp = strsplit(sub_fold,'_');
    rid = tmp{4};
    sid = strcat('s',rid);
    % Only grab the file if subject passed QC
    if any(ismember(list_subject,sid))
        files_in.data.network_1.(sid) = [path_data filesep sub_fold];
    end
end
%% options
opt.folder_out = path_results;
opt.scale = 1;
opt.stack.regress_conf = {'gender','age_scan','mean_gm','tiv'};

opt.subtype.nb_subtype = scale;

%% run the pipeline
%opt.psom.qsub_options = '--mem=8G  --account rrg-pbellec --time=00-03:00 --ntasks=1 --cpus-per-task=1  ';

opt.flag_test = false;
[pipe,opt] = niak_pipeline_subtype(files_in,opt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% extract weights in CN subjects from ADNI2
clear all

path_data = '/home/atam/adni_vbm_octave/adni2/';
path_model = '/home/atam/adni_vbm_octave/adni2_bl_vbm_model_niak.csv';
path_mask = '/home/atam/yasser_templates/rc1mni_icbm152_t1_tal_nlin_asym_09c_6.nii';  
path_out = '/home/atam/adni1_vbm_adcn_subtypes_20171209/';

scale = 7;

path_results = strcat(path_out, num2str(scale), 'clus/', 'adni2_CN_weights');

%% set up files_in structure

files_in.mask = path_mask;
files_in.model = path_model;
files_in.subtype.network_1 = strcat('/home/atam/adni1_vbm_adcn_subtypes_20171209/',num2str(scale),'clus/networks/network_1/subtype_network_1.mat');

%% filter out those with failed QC in model
[conf_model,list_subject,cat_names] = niak_read_csv(files_in.model);
qc_col = find(strcmp('vbm_qc',cat_names));
mask_qc = logical(conf_model(:,qc_col));
conf_model = conf_model(mask_qc,:);
list_subject = list_subject(mask_qc);

%% filter just the AD patients
cn_col = find(strcmp('CN',cat_names));
nan_cn = isnan(conf_model(:,cn_col));
% mask out the nan values
conf_model = conf_model(~nan_cn,:);
list_subject = list_subject(~nan_cn);
% find mci subjects and mask everyone else out
mask_cn = logical(conf_model(:,cn_col));
conf_model = conf_model(mask_cn,:);
list_subject = list_subject(mask_cn);

%% grab oldest mwrc1rl_T1 from each subject's folder

folds = dir(path_data);
folds = {folds.name};
folds = folds(~ismember(folds,{'.','..','logs','qc','models'}));

for ss = 1:length(folds)
    % From folder name, grab subject ID
    sub_fold = folds{ss};
    tmp = strsplit(sub_fold,'_');
    rid = tmp{4};
    sid = strcat('s',rid);
    % Only grab the file if subject passed QC
    if any(ismember(list_subject,sid))
        files_in.data.network_1.(sid) = [path_data filesep sub_fold];
    end
end
%% options
opt.folder_out = path_results;
opt.scale = 1;
opt.stack.regress_conf = {'gender','age_scan','mean_gm','tiv'};

opt.subtype.nb_subtype = scale;

%% run the pipeline
%opt.psom.qsub_options = '--mem=8G  --account rrg-pbellec --time=00-03:00 --ntasks=1 --cpus-per-task=1  ';

opt.flag_test = false;
[pipe,opt] = niak_pipeline_subtype(files_in,opt);

%%%%%%%%%%%%%%%%%%%
%% weight extraction in prevent-ad

clear all

path_data = '/home/atam/preventad_vbm_bl_octave/vbm_data/';
path_model = '/home/atam/preventad_vbm_bl_octave/preventad_model_dr5_20171212_niak.csv';
path_mask = '/home/atam/yasser_templates/rc1mni_icbm152_t1_tal_nlin_asym_09c_6.nii';  
path_out = '/home/atam/adni1_vbm_adcn_subtypes_20171209/';

scale = 7;

path_results = strcat(path_out, num2str(scale), 'clus/', 'pad_weights');

%% set up files_in structure

files_in.mask = path_mask;
files_in.model = path_model;
files_in.subtype.network_1 = strcat('/home/atam/adni1_vbm_adcn_subtypes_20171209/',num2str(scale),'clus/networks/network_1/subtype_network_1.mat');

%% filter out those with failed QC in model
[conf_model,list_subject,cat_names] = niak_read_csv(files_in.model);
qc_col = find(strcmp('vbm_qc',cat_names));
mask_qc = logical(conf_model(:,qc_col));
conf_model = conf_model(mask_qc,:);
list_subject = list_subject(mask_qc);

%% grab oldest mwrc1rl_T1 from each subject's folder

folds = dir(path_data);
folds = {folds.name};
folds = folds(~ismember(folds,{'.','..','logs','qc','models'}));

for ss = 1:length(folds)
    % From folder name, grab subject ID
    sub_fold = folds{ss};
    tmp = strsplit(sub_fold,'_');
    rid = tmp{3};
    sid = strcat('s',rid);
    % Only grab the file if subject passed QC
    if any(ismember(list_subject,sid))
        files_in.data.network_1.(sid) = [path_data filesep sub_fold];
    end
end
%% options
opt.folder_out = path_results;
opt.scale = 1;
opt.stack.regress_conf = {'sex','Candidate_Age','mean_gm','tiv'};

opt.subtype.nb_subtype = scale;

%% run the pipeline
%opt.psom.qsub_options = '--mem=8G  --account rrg-pbellec --time=00-03:00 --ntasks=1 --cpus-per-task=1  ';

opt.flag_test = false;
[pipe,opt] = niak_pipeline_subtype(files_in,opt);

