function [pipeline, opt] = niak_spm_pipeline_vbm(files_in,opt)
% Pipeline for running VBM with SPM
% 
% SYNTAX: 
% [PIPELINE, OPT] = NIAK_SPM_PIPELINE_VBM(FILES_IN, OPT)
% _________________________________________________________________________
%
% INPUTS:
% 
% FILES_IN (structure) with the following fields:
% 
%     ANAT.<SUBJECT> 
%       (string) raw anatomical (T1-weighted MR) volume.
%     
%     DARTEL_TEMPLATE 
%       (structure, optional) with the following fields:
%       TEMP_1 (string) full path and filename for an existing DARTEL template
%       TEMP_2 (string) full path and filename for an existing DARTEL template
%       TEMP_3 (string) full path and filename for an existing DARTEL template
%       TEMP_4 (string) full path and filename for an existing DARTEL template
%       TEMP_5 (string) full path and filename for an existing DARTEL template
%       TEMP_6 (string) full path and filename for an existing DARTEL template
%       If specified, the resulting segmentations will be warped to match 
%       these templates. Early iterations should ideally use smoother 
%       templates and more regularisation than later iterations (e.g. 
%       TEMP_1 should be smoother than TEMP_6).
%
%    STEREO_TEMPLATE
%       (structure, optional) with the following fields:
%       TEMPLATE (string, default '') the file name of the target template. If
%           left empty, the default will be used, i.e. the MNI-152
%           symmetrical non-linear average. 
%       TEMPLATE_MASK (string, default '') the file name of a binary mask of 
%           a region of interest in the template space. If left empty, the 
%           default will be used, i.e. a brain mask of the default template
%       These templates will be used to perform a linear registration of the 
%       T1 anatomical scans in native space to the MNI stereotaxis space.
%
% OPT (structure) with the following fields:
% 
%     ANAT2STEREOLIN 
%       (structure) the options of the NIAK_BRICK_ANAT2STEREOLIN
%       function. Defaults should work.
%
%     FOLDER_OUT 
%       (string) where to write the results of the pipeline.
%
%     FLAG_TEST
%       (boolean, default false) If FLAG_TEST is true, the pipeline
%       will just produce a pipeline structure, and will not actually
%       process the data. Otherwise, PSOM_RUN_PIPELINE will be used to
%       process the data.
%
%     PSOM
%       (structure) the options of the pipeline manager. See the OPT
%       argument of PSOM_RUN_PIPELINE. Default values can be used here.
%       Note that the field PSOM.PATH_LOGS will be set up by the
%       pipeline.
% _________________________________________________________________________
% OUTPUTS: 
%
%   PIPELINE 
%       (structure) describe all jobs that need to be performed in the
%       pipeline.
%
% _________________________________________________________________________
% Copyright (c) Angela Tam, Yasser Iturria-Medina, Pierre Bellec
% Montreal Neurological Institute, 2017
% Centre de recherche de l'institut de geriatrie de Montreal,
% Department of Computer Science and Operations Research
% University of Montreal, Quebec, Canada, 2017
% Maintainer : pierre.bellec@criugm.qc.ca
% See licensing information in the code.
% Keywords : SPM, DARTEL, nonlinear registration, interface, NIAK

% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.

%% import NIAK global variables
niak_gb_vars

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Seting up default arguments %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Input files
if ~exist('files_in','var')||~exist('opt','var')
    error('niak:brick','syntax: PIPELINE = NIAK_SPM_PIPELINE_VBM(FILES_IN,OPT).\n Type ''help niak_spm_pipeline_vbm'' for more info.')
end

%% Checking that FILES_IN is in the correct format
if ~isstruct(files_in)

    error('FILES_IN should be a struture!')
    
else
   
    list_subject = fieldnames(files_in.anat);
    nb_subject = length(list_subject);
    list_anat = cell([length(list_subject) 1]);
    
    for num_s = 1:nb_subject
        
        subject = list_subject{num_s};
        data_subject = files_in.anat.(subject);
        
        if ~ischar(data_subject)
            error('FILES_IN.%s should be a string!',upper(subject));
        end
        
        list_anat{num_s} = files_in.anat.(subject);        
    end
    
end

%% Files_in
files_in = psom_struct_defaults(files_in,...
          { 'anat' , 'dartel_template' , 'stereo_template' },...
           { NaN    , struct([])        , struct()          });
opt = psom_struct_defaults(opt , ...
           { 'flag_test' , 'folder_out' , 'flag_verbose' , 'psom'   , 'anat2stereolin' },...
           { false       , NaN          , true           , struct() , struct()          });

opt.folder_out = niak_full_path(opt.folder_out);
opt.psom.path_logs = [opt.folder_out 'logs' filesep];

%% Pipeline starts here

pipeline = [];

for num_s = 1:nb_subject

   
   
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%   CROP NECK    %%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    clear files_in_tmp files_out_tmp opt_tmp
    name_brick      = 'niak_brick_crop_neck'; 
    subject         = list_subject{num_s};
    name_job        = ['crop_neck_' subject];   

    % Files in
    files_in_tmp              = list_anat{num_s};

    files_out_tmp = [opt.folder_out filesep subject filesep subject '_T1w_cropped.nii'];
    opt_tmp.crop_neck    = 0.25;


    pipeline = psom_add_job(pipeline,name_job,name_brick,files_in_tmp,files_out_tmp,opt_tmp);   
    	
    % Get filename for linear registration
    files_in_coreg = pipeline.(name_job).files_out;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Run a linear coregistration in stereotaxic space %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Names
    clear files_in_tmp files_out_tmp opt_tmp
    name_brick      = 'niak_brick_anat2stereolin';    
    subject         = list_subject{num_s};
    name_job        = ['t1_stereolin_' subject];
    
    % Files in
    files_in_tmp.t1              = files_in_coreg;
    files_in_tmp.t1_mask         = 'gb_niak_omitted';
    % Building default input names for template
    if isfield(files_in.stereo_template, 'template')
        files_in_tmp.template = files_in.stereo_template.template;
    else
        files_in_tmp.template        = '';
    end
    if isfield(files_in.stereo_template, 'template_mask')
        files_in_tmp.template_mask = files_in.stereo_template.template_mask;
    else
        files_in_tmp.template_mask   = '';
    end
    % Files out
    files_out_tmp.transformation = '';
    files_out_tmp.t1_stereolin   = '';
    % Options
    opt_tmp                      = opt.anat2stereolin;
    opt_tmp.flag_test            = opt.flag_test;
    opt_tmp.flag_verbose         = opt.flag_verbose;
    opt_tmp.folder_out = [opt.folder_out filesep subject];
    

   
    % Add job
    pipeline = psom_add_job(pipeline,name_job,name_brick,files_in_tmp,files_out_tmp,opt_tmp);   
    
    % Get names for AC-PC alignment
    files_in_acpc = pipeline.(name_job).files_out.t1_stereolin;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Align to AC-PC                                   %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Names
    clear files_in_tmp files_out_tmp opt_tmp
    name_brick      = 'spm_brick_comm_adjust';
    subject         = list_subject{num_s};
    name_job        = ['align_acpc_' subject];
    
    % Files in
    files_in_tmp                = files_in_acpc;
    files_out_tmp               = '';
    opt_tmp.flag_test           = false;
    opt_tmp.folder_out          = [opt.folder_out filesep subject];
    
    % Add job
    pipeline = psom_add_job(pipeline,name_job,name_brick,files_in_tmp,files_out_tmp,opt_tmp);
    
    % Get names for segmentation
    files_in_seg = pipeline.(name_job).files_out;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Perform the segmentation                         %%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % Names
    clear files_in_tmp files_out_tmp opt_tmp
    name_brick      = 'spm_brick_preproc';
    subject         = list_subject{num_s};
    name_job        = ['segment_' subject];
    
    % Files in
    files_in_tmp                = files_in_seg;
    files_out_tmp.gm_pve        = '';
    files_out_tmp.wm_pve        = '';
    files_out_tmp.csf_pve       = '';
    files_out_tmp.gm_pve_r      = '';
    files_out_tmp.wm_pve_r      = '';
    files_out_tmp.csf_pve_r     = '';
    files_out_tmp.def           = '';
    files_out_tmp.inv_def       = '';
    files_out_tmp.prov          = '';
    opt_tmp.flag_test           = opt.flag_test;
    opt_tmp.folder_out          = '';
    
    % Add job
    pipeline = psom_add_job(pipeline,name_job,name_brick,files_in_tmp,files_out_tmp,opt_tmp);
    
    % Get names for DARTEL
    files_in_dartel.gm.(subject) = pipeline.(name_job).files_out.gm_pve_r;
    files_in_dartel.wm.(subject) = pipeline.(name_job).files_out.wm_pve_r;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DARTEL                                           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear files_in_tmp files_out_tmp opt_tmp

if isempty(files_in.dartel_template)
    %%%%%%%%%%%%%%%%%%%
    %% Create templates
    %%%%%%%%%%%%%%%%%%%
    name_brick      = 'spm_brick_dartel_template';
    name_job        = 'dartel_template';
    
    % files in
    files_in_tmp                = files_in_dartel;
    files_out_tmp.template      = '';
    files_out_tmp.flow_field    = struct;
    opt_tmp.flag_test           = opt.flag_test;
    opt_tmp.folder_out          = '';
    
    % add job
    pipeline = psom_add_job(pipeline,name_job,name_brick,files_in_tmp,files_out_tmp,opt_tmp); 
    
    % get filenames for next brick
    for num_s = 1:nb_subject
        files_in_jac.(list_subject{num_s}) = pipeline.(name_job).files_out.flow_field.(list_subject{num_s}); 
    end
else
    %%%%%%%%%%%%%%%%%%%%%
    %% Register to existing templates for each subject
    %%%%%%%%%%%%%%%%%%%%%
    for num_s = 1:nb_subject
    clear files_in_tmp files_out_tmp opt_tmp
        % names
        name_brick      = 'spm_brick_dartel_warp';
        subject         = list_subject{num_s};
        name_job        = ['dartel_warp_' subject];
        
        % files in
        files_in_tmp.rc_img.gm.(subject)    = files_in_dartel.gm.(subject);
        files_in_tmp.rc_img.wm.(subject)    = files_in_dartel.wm.(subject);
        files_in_tmp.template.temp_1        = files_in.dartel_template.temp_1;
        files_in_tmp.template.temp_2        = files_in.dartel_template.temp_2;
        files_in_tmp.template.temp_3        = files_in.dartel_template.temp_3;
        files_in_tmp.template.temp_4        = files_in.dartel_template.temp_4;
        files_in_tmp.template.temp_5        = files_in.dartel_template.temp_5;
        files_in_tmp.template.temp_6        = files_in.dartel_template.temp_6;
        files_out_tmp                       = struct;
        opt_tmp.flag_test                   = opt.flag_test;
        opt_tmp.folder_out                  = '';
        
        % add job
        pipeline = psom_add_job(pipeline,name_job,name_brick,files_in_tmp,files_out_tmp,opt_tmp);
        
        % get filename for next brick
        files_in_jac.(subject) = pipeline.(name_job).files_out.(subject);
        
    end
end

for num_s = 1:nb_subject
    %%%%%%%%%%%%%%%%%
    %% Save jacobians   
    %%%%%%%%%%%%%%%%%
    clear files_in_tmp files_out_tmp opt_tmp
    % names
    name_brick      = 'spm_brick_dartel_jacobian';
    subject         = list_subject{num_s};
    name_job        = ['dartel_jacobian_' subject];
    
    % files in
    [path_r, name_r, ext_r] = fileparts(files_in_dartel.gm.(subject));
    if isempty(path_r)
        path_r = '.';
    end
%     flowfield               = [path_r filesep ff_pre name_r ext_r];
    files_in_tmp.(subject)  = files_in_jac.(subject);
    files_out_tmp           = struct;
    opt_tmp.flag_test       = opt.flag_test;
    opt_tmp.folder_out      = '';
    
    % add job to pipeline
    pipeline = psom_add_job(pipeline,name_job,name_brick,files_in_tmp,files_out_tmp,opt_tmp);
    
    %%%%%%%%%%%%%%%%%%
    %% Warp individual images to template and smooth
    %%%%%%%%%%%%%%%%%%
    clear files_in_tmp files_out_tmp opt_tmp
    % names
    name_brick      = 'spm_brick_dartel_norm';
    subject         = list_subject{num_s};
    name_job        = ['dartel_norm_' subject];
    
    % files in
    [path_r, name_r, ext_r] = fileparts(files_in_dartel.gm.(subject));
    if isempty(path_r)
        path_r = '.';
    end
    files_in_tmp.images.(subject)       = files_in_dartel.gm.(subject);
    files_in_tmp.flowfields.(subject)   = files_in_jac.(subject);
    files_out_tmp                       = struct;
    opt_tmp.flag_test                   = opt.flag_test;
    opt_tmp.folder_out                  = '';
    
    % add job to pipeline
    pipeline = psom_add_job(pipeline,name_job,name_brick,files_in_tmp,files_out_tmp,opt_tmp);
end

%%%%%%%%%%%%%%%%%%%%%%%
%% Run the pipeline! %%
%%%%%%%%%%%%%%%%%%%%%%%
if ~opt.flag_test
    psom_run_pipeline(pipeline,opt.psom);
end


