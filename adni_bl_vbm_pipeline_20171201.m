%% scripts to run dartel on baseline adni scans on cedar

%% adni2 baseline scans
clear all

addpath(genpath('/home/atam/spm12'))
addpath(genpath('/home/atam/git/spm_container'))
path_temp = '/home/atam/projects/rrg-pbellec/PREPROCESS/adni/T1/yasser_templates';
path_data = '/home/atam/scratch/raw_adni2_bl';

% make files in structure
files_in = struct;
list_sub = dir(path_data);
list_sub = {list_sub.name};
list_sub = list_sub(~ismember(list_sub,{'.','..'}));

for ss = 1:length(list_sub)
    tmp = strsplit(list_sub{ss},'_');
    rid = tmp{3};
    % take the first (baseline) file
    sub_files = dir([path_data filesep list_sub{ss}]);
    sub_files = {sub_files.name};
    sub_files = sub_files(~ismember(sub_files,{'.','..'}));
    bl_scan = sub_files{1};
    sub_name = strcat('s', rid);
    files_in.anat.(sub_name) = [path_data filesep list_sub{ss} filesep bl_scan];
end

files_in.dartel_template.temp_1 = [path_temp filesep 'template_1.nii'];
files_in.dartel_template.temp_2 = [path_temp filesep 'template_2.nii'];
files_in.dartel_template.temp_3 = [path_temp filesep 'template_3.nii'];
files_in.dartel_template.temp_4 = [path_temp filesep 'template_4.nii'];
files_in.dartel_template.temp_5 = [path_temp filesep 'template_5.nii'];
files_in.dartel_template.temp_6 = [path_temp filesep 'template_6.nii'];

opt.folder_out = '/home/atam/scratch/adni2_bl_vbm_dartel_20171201';
opt.psom.qsub_options = '--mem=8G  --account rrg-pbellec --time=00-30:00 --ntasks=1 --cpus-per-task=1  ';

[pipe,opt] = niak_spm_pipeline_vbm(files_in,opt);

%% adni1 baseline scans
clear all

addpath(genpath('/home/atam/spm12'))
addpath(genpath('/home/atam/git/spm_container'))
path_temp = '/home/atam/projects/rrg-pbellec/PREPROCESS/adni/T1/yasser_templates';
path_data = '/home/atam/scratch/raw_adni1_bl';

% make files in structure
files_in = struct;
list_sub = dir(path_data);
list_sub = {list_sub.name};
list_sub = list_sub(~ismember(list_sub,{'.','..'}));

for ss = 1:length(list_sub)
    tmp = strsplit(list_sub{ss},'_');
    rid = tmp{3};
    % take the first folder 
    fold1 = dir([path_data filesep list_sub{ss}]);
    fold1 = {fold1.name};
    fold1 = fold1(~ismember(fold1,{'.','..'}));
    fold1 = fold1{1};
    % get subfolder name
    fold2 = dir([path_data filesep list_sub{ss} filesep fold1]);
    fold2 = {fold2.name};
    fold2 = fold2(~ismember(fold2,{'.','..'}));
    fold2 = fold2{1};
    % get subfolder name
    fold3 = dir([path_data filesep list_sub{ss} filesep fold1 filesep fold2]);
    fold3 = {fold3.name};
    fold3 = fold3(~ismember(fold3,{'.','..'}));
    fold3 = fold3{1};
    % get full path of desired scan
    path_scan = [path_data filesep list_sub{ss} filesep fold1 filesep fold2 filesep fold3];
    sub_files = dir(path_scan);
    sub_files = {sub_files.name};
    sub_files = sub_files(~ismember(sub_files,{'.','..'}));
    bl_scan = sub_files{1};
    sub_name = strcat('s', rid);
    files_in.anat.(sub_name) = [path_scan filesep bl_scan];
end

files_in.dartel_template.temp_1 = [path_temp filesep 'template_1.nii'];
files_in.dartel_template.temp_2 = [path_temp filesep 'template_2.nii'];
files_in.dartel_template.temp_3 = [path_temp filesep 'template_3.nii'];
files_in.dartel_template.temp_4 = [path_temp filesep 'template_4.nii'];
files_in.dartel_template.temp_5 = [path_temp filesep 'template_5.nii'];
files_in.dartel_template.temp_6 = [path_temp filesep 'template_6.nii'];

opt.folder_out = '/home/atam/scratch/adni1_bl_vbm_dartel_20171202';
opt.psom.qsub_options = '--mem=8G  --account rrg-pbellec --time=00-30:00 --ntasks=1 --cpus-per-task=1  ';

[pipe,opt] = niak_spm_pipeline_vbm(files_in,opt);

