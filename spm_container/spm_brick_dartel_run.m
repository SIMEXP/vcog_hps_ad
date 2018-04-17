function [in, out, opt] = spm_brick_dartel_run(in,out,opt)

% This function will generate a series of template images by iteratively 
% computing a templates with mean shape and intensities and flow fields 
% and/or register images to template data. 

% IN (structure) with the following fields:
%   RC_IMG (1 x 2 cell array) Contains filenames for images to be registered.
%       The first cell must contain filenames for the imported GREY MATTER
%       images (rc1*.nii) and the second cell must contain filenames for
%       the imported WHITE MATTER images (rc2*.nii). The order must be the
%       same across both cells, such that the grey matter image for any
%       subject corresponds with the appropriate white matter image.
%       Each cell must be an N x 1 cell, where N equals the number of
%       images, and each sub-cell contains the filename for ONE image.
%   TEMPLATE (structure, optional) with the following fields:
%       TEMP_1 (string) full path and filename for an existing template
%       TEMP_2 (string) full path and filename for an existing template
%       TEMP_3 (string) full path and filename for an existing template
%       TEMP_4 (string) full path and filename for an existing template
%       TEMP_5 (string) full path and filename for an existing template
%       TEMP_6 (string) full path and filename for an existing template
%       If specified, the images in IN.ANAT will be warped to match these 
%       templates. Early iterations should ideally use smoother templates 
%       and more regularisation than later iterations (e.g. TEMP_1 should
%       be smoother than TEMP_6).
%       
% OUT (structure) with the following fields:
%   TEMPLATE (string, default 'Template') base name of the resulting
%       templates
%   FLOW_FIELD (string, default 'u_rc1') prefix of the resulting
%       flow fields
%   JAC (string, default 'jac_') prefix of the resulting Jacobian
%       determinant fields
%   GM_WARPED (string, default 'mw') prefix of the resulting grey
%       matter images warped to template space
%   SMOOTH (string, default 's') prefix of the resulting smoothed warped grey
%       matter images
%
% OPT.FOLDER_OUT (string, default same as IN.RC_IMG{1}{N}) the folder where 
%       to generate outputs. If left empty, the outputs for a given image
%       in IN.RC_IMG{1} will be written to the same originating path (i.e.
%       path of IN.RC_IMG{1}{1} == path of outputs from IN.RC_IMG{1}{1}).
% OPT.FLAG_TEST (boolean, default false) flag to run a test without 
%       generating outputs.
%
% If OPT.FLAG_TEST is true, the brick does nothing but still updates the
%    default file names in OUT, as well as default options in OPT.
% By default (OUT = struct) the brick generates all output files.
%    If file names are specified in OUT, those will be used over defaults.
%    If a file name 'skip' is specified, the output is not generated.
%
% This is an interface to the DARTEL tools from SPM12.
% It performs non-linear registration. This involves iteratively matching 
% all the selected images to a template generated from their own mean. A 
% series of Template*.nii files are generated, which become increasingly 
% crisp as the registration proceeds. The non-linear registration can also 
% match individual images to pre-existing template data. Start out with 
% smooth templates, and select crisp templates for the later iterations.
%
% _________________________________________________________________________
% Copyright (c) Angela Tam, Yasser Iturria-Medina, Pierre Bellec
% Montreal Neurological Institute, 2017
% Centre de recherche de l'institut de geriatrie de Montreal,
% Department of Computer Science and Operations Research
% University of Montreal, Quebec, Canada, 2017
% Maintainer : pierre.bellec@criugm.qc.ca
% See licensing information in the code.
% Keywords : SPM, DARTEL, nonlinear registration, interface

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
global gb_psom_name_job

if ~exist('in','var')||~exist('out','var')
    error('spm:brick','syntax: [IN,OUT,OPT] = SPM_BRICK_DARTEL_RUN(IN,OUT,OPT).\n Type ''help spm_brick_dartel_run'' for more info.')
end

%% Input
if ~isstruct(in)
  error('IN should be a structure')
end
[path_f,~,ext_f] = fileparts(in.rc_img{1}{1}); 
if isempty(path_f)
    path_f = '.';
end

%% Temporary folder
if ~isempty(gb_psom_name_job)
    path_tmp = [tempdir filesep gb_psom_name_job filesep];
else
    path_tmp = tempdir;
end

%% Files_in
in = psom_struct_defaults(in,...
        { 'rc_img' , 'template' },...
        { NaN      , struct([]) });

%% Options
if nargin < 2
  out = struct;
end
out = psom_struct_defaults(out , ...
        { 'template' , 'flow_field', 'jac', 'gm_warped', 'smooth' } , ...
        { ''         , ''          , ''   , ''         , ''       });

%% Options
if nargin < 3
  opt = struct;
end
opt = psom_struct_defaults(opt , ...
        { 'flag_test', 'folder_out' },...
        { false      , ''           });

%% settings for DARTEL

opt_dartel.settings.rform = 0;
opt_dartel.settings.param(1).its = 3;
opt_dartel.settings.param(1).rparam = [4 2 1e-06];
opt_dartel.settings.param(1).K = 0;
opt_dartel.settings.param(2).its = 3;
opt_dartel.settings.param(2).rparam = [2 1 1e-06];
opt_dartel.settings.param(2).K = 0;
opt_dartel.settings.param(3).its = 3;
opt_dartel.settings.param(3).rparam = [1 0.5 1e-06];
opt_dartel.settings.param(3).K = 1;
opt_dartel.settings.param(4).its = 3;
opt_dartel.settings.param(4).rparam = [0.5 0.25 1e-06];
opt_dartel.settings.param(4).K = 2;
opt_dartel.settings.param(5).its = 3;
opt_dartel.settings.param(5).rparam = [0.25 0.125 1e-06];
opt_dartel.settings.param(5).K = 4;
opt_dartel.settings.param(6).its = 3;
opt_dartel.settings.param(6).rparam = [0.25 0.125 1e-06];
opt_dartel.settings.param(6).K = 6;
opt_dartel.settings.optim.lmreg = 0.01;
opt_dartel.settings.optim.cyc = 3;
opt_dartel.settings.optim.its = 3;
opt_dartel.folder_out = '';
opt_dartel.flag_test = opt.flag_test;

opt = psom_defaults(opt_dartel, opt );

if strcmp(opt.folder_out,'')
    opt.folder_out = path_f;
end

%% Building default output names
if isempty(in.template) && isempty(out.template)
    out.template = 'Template';
end
if isempty(out.flow_field)
    out.flow_field = 'u_';
end
if isempty(out.jac)
    out.jac = 'jac_';
end
if isempty(out.gm_warped)
    out.gm_warped = 'mw';
end
if isempty(out.smooth)
    out.smooth = 's';
end

if opt.flag_test == 1
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Brick really starts here
%%%%%%%%%%%%%%%%%%%%%%%%%%%
opt.images = in.rc_img;
folder_out = opt.folder_out;
opt = rmfield ( opt , { 'folder_out' , 'flag_test'}); % Get rid of the options not supported by spm dartel functions

%% create new templates
if isempty(in.template)
    opt.settings.template = out.template;
    opt.settings.param(1).slam = 16;
    opt.settings.param(2).slam = 8;
    opt.settings.param(3).slam = 4;
    opt.settings.param(4).slam = 2;
    opt.settings.param(5).slam = 1;
    opt.settings.param(6).slam = 0.5;
    spm_dartel_template(opt);
else
    %% or register to existing templates
    opt.settings.param(1).template = cellstr(in.template.temp_1);
    opt.settings.param(2).template = cellstr(in.template.temp_2);
    opt.settings.param(3).template = cellstr(in.template.temp_3);
    opt.settings.param(4).template = cellstr(in.template.temp_4);
    opt.settings.param(5).template = cellstr(in.template.temp_5);
    opt.settings.param(6).template = cellstr(in.template.temp_6);
    spm_dartel_warp(opt);
end

%% saving the determinants of the jacobians
j_opt = struct;
j_opt.flowfields = cell(size(in.rc_img{1},1),1); 
for ff = 1:size(in.rc_img{1},1)
    [path_r,name_r,ext_r] = fileparts(in.rc_img{1}{ff});
    f_name = ['u_' name_r ext_r];
    j_opt.flowfields{ff} = [path_r filesep f_name];
end
j_opt.K = 6;
spm_dartel_jacobian(j_opt);

%% warp to group-specific template space
w_opt = struct;
w_opt.images = {in.rc_img{1}};
w_opt.jactransf = 1;
w_opt.K = 6;
w_opt.interp = 7;
w_opt.flowfields = j_opt.flowfields;
spm_dartel_norm(w_opt);

%% smooth the warped images
s_opt = struct;
s_opt.fwhm = [8 8 8];
s_opt.dtype = 0;
for ff = 1:(size(in.c1_img{1},1))
    [path_r,name_r,ext_r] = fileparts(in.rc_img{1}{ff});
    s_in = [path_r filesep 'mw' name_r ext_r];
    s_out = [path_r filesep out.smooth 'mw' name_r ext_r];
    spm_smooth(s_in, s_out, s_opt.fwhm, s_opt.dtype);
end

%% rename the flow_field outputs to user-specified filenames
if ~strcmp(out.flow_field, 'u_')
    for ff = 1:size(j_opt.flowfields,1)
        nname = strrep(j_opt.flowfields{ff}, 'u_', out.flow_field);
        movefile(j_opt.flowfields{ff}, nname);
    end
end

%% rename the jacobian outputs to user-specified filenames
if ~strcmp(out.jac, 'jac_')
    for ff = 1:size(in.rc_img{1},1)
        [path_r,name_r,ext_r] = fileparts(in.rc_img{1}{ff});
        jac = [path_r filesep 'jac_' name_r ext_r];
        nname = strrep(jac, 'jac_', out.jac);
        movefile(jac, nname);
    end
end

%% rename the warped images outputs to user-specified filenames
if ~strcmp(out.gm_warped, 'mw')
    for ff = 1:size(in.rc_img{1},1)
        [path_r,name_r,ext_r] = fileparts(in.rc_img{1}{ff});
        mw_file = [path_r filesep 'mw' name_r ext_r];
        nname = strrep(mw_file, 'mw', out.gm_warped);
        movefield(mw_file, nname);
    end
end

%% move the outputs to opt.folder_out if specified
if ~strcmp(folder_out, path_f)
    if ~isempty(out.template)
        [path_r,name_r,ext_r] = fileparts(in.rc_img{1}{1});
        movefile([path_r filesep out.template '*' ext_r], folder_out);
    end
end

if ~strcmp(folder_out, path_f)
    for ff = 1:size(in.rc_img{1},1)
        [path_r,name_r,ext_r] = fileparts(in.rc_img{1}{ff});
        movefile([j_opt.flowfields{ff}], folder_out);
        movefile([path_r filesep out.jac name_r ext_r], folder_out);
        movefile([path_r filesep out.gm_warped name_r ext_r], folder_out);
        movefile([path_r filesep out.smooth out.gm_warped name_r ext_r], folder_out);
    end
end

        
      


    

