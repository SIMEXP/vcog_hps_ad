function [in, out, opt] = spm_brick_dartel_template(in,out,opt)

% This function will generate a series of template images by iteratively 
% computing a templates with mean shape and intensities and flow fields 
% and register images to the new template data. 
%
% IN (structure) withe the following fields:
%       GM.<SUBJECT> (string) the imported grey matter image
%       WM.<SUBJECT> (string) the imported white matter image
%       The subject order for the fields GM and WM must be the same, such
%       that the grey matter image for any subject corresponds with the
%       appropriate white matter image.
%       
% OUT (structure) with the following fields:
%       TEMPLATE (string, default 'Template') base name of the resulting
%           templates
%       FLOW_FIELD.<SUBJECT> (string, default ['u_' GM.<SUBJECT>) filename 
%           of the resulting flow fields
%
% OPT.FOLDER_OUT (string, default same as IN.GM.<SUBJECT>) the folder where 
%       to generate outputs. If left empty, the outputs for a given image
%       in IN.GM.<SUBJECT> will be written to the same originating path (i.e.
%       path of IN.GM.<SUBJECT> == path of outputs from IN.GM.<SUBJECT>).
%
% OPT.FLAG_TEST (boolean, default false) flag to run a test without 
%       generating outputs.
% If OPT.FLAG_TEST is true, the brick does nothing but still updates the
%       default file names in OUT, as well as default options in OPT.
%       By default (OUT = struct) the brick generates all output files.
%       If file names are specified in OUT, those will be used over defaults.
%       If a file name 'skip' is specified, the output is not generated.
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
    error('spm:brick','syntax: [IN,OUT,OPT] = SPM_BRICK_DARTEL_TEMPLATE(IN,OUT,OPT).\n Type ''help spm_brick_dartel_template'' for more info.')
end

%% Input
if ~isstruct(in)
  error('IN should be a structure')
end
list_sub = fieldnames(in.gm);
[path_f,~,ext_f] = fileparts(in.gm.(list_sub{1})); 
if isempty(path_f)
    path_f = '.';
end

%% Temporary folder
if ~isempty(gb_psom_name_job)
    path_tmp = [tempdir filesep gb_psom_name_job filesep];
else
    path_tmp = tempdir;
end

%% Options
if nargin < 2
  out = struct;
end
out = psom_struct_defaults(out , ...
        { 'template' , 'flow_field' } , ...
        { ''         , struct       });

%% Options
if nargin < 3
  opt = struct;
end
opt = psom_struct_defaults(opt , ...
        { 'flag_test', 'folder_out' },...
        { false      , ''           });
    
%% Building default output names
if isempty(out.template)
    out.template = 'Template';
end

list_sub = fieldnames(in.gm);
if isempty(out.flow_field) || isempty(fieldnames(out.flow_field))
    for ss = 1:length(list_sub)
        [path_f,name_f,ext_f] = fileparts(in.gm.(list_sub{ss}));
        if isempty(path_f)
            path_f = '.';
        end
        out.flow_field.(list_sub{ss}) = [path_f filesep 'u_' name_f '_Template' ext_f];
    end
end


%% settings for DARTEL

opt_dartel.settings.rform = 0;
opt_dartel.settings.param(1).its = 3;
opt_dartel.settings.param(1).rparam = [4 2 1e-06];
opt_dartel.settings.param(1).K = 0;
opt_dartel.settings.param(1).slam = 16;
opt_dartel.settings.param(2).its = 3;
opt_dartel.settings.param(2).rparam = [2 1 1e-06];
opt_dartel.settings.param(2).K = 0;
opt_dartel.settings.param(2).slam = 8;
opt_dartel.settings.param(3).its = 3;
opt_dartel.settings.param(3).rparam = [1 0.5 1e-06];
opt_dartel.settings.param(3).K = 1;
opt_dartel.settings.param(3).slam = 4;
opt_dartel.settings.param(4).its = 3;
opt_dartel.settings.param(4).rparam = [0.5 0.25 1e-06];
opt_dartel.settings.param(4).K = 2;
opt_dartel.settings.param(4).slam = 2;
opt_dartel.settings.param(5).its = 3;
opt_dartel.settings.param(5).rparam = [0.25 0.125 1e-06];
opt_dartel.settings.param(5).K = 4;
opt_dartel.settings.param(5).slam = 1;
opt_dartel.settings.param(6).its = 3;
opt_dartel.settings.param(6).rparam = [0.25 0.125 1e-06];
opt_dartel.settings.param(6).K = 6;
opt_dartel.settings.param(6).slam = 0.5;
opt_dartel.settings.optim.lmreg = 0.01;
opt_dartel.settings.optim.cyc = 3;
opt_dartel.settings.optim.its = 3;
opt_dartel.folder_out = '';
opt_dartel.flag_test = opt.flag_test;

opt = psom_defaults(opt_dartel, opt );

if opt.flag_test == 1
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Brick really starts here
%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Convert the files_in structure to spm-friendly inputs

spm_in = cell(1,2);
spm_in{1} = cell(size(list_sub,1),1);
spm_in{2} = cell(size(list_sub,1),1);
spm_in{1} = struct2cell(in.gm);
spm_in{2} = struct2cell(in.wm);

opt.images = spm_in;
opt.settings.template = out.template;
folder_out = opt.folder_out;
opt = rmfield ( opt , { 'folder_out' , 'flag_test'}); % Get rid of the options not supported by spm dartel functions

%% create new templates
spm_dartel_template(opt);

%% rename the flow_field outputs to user-specified filenames
for ss = 1:length(list_sub)
    f_out = out.flow_field.(list_sub{ss});
    [path_r,name_r,ext_r] = fileparts(spm_in{1}{ss});
    if isempty(path_r)
        path_r = '.';
    end
    flowfield = [path_r filesep 'u_' name_r '_Template' ext_r];
    if ~strcmp(f_out, flowfield)
        movefile(flowfield, f_out);
    end
end

%% move the outputs to opt.folder_out if specified
if ~strcmp(folder_out, path_f)
    if ~isempty(out.template)
        movefile([path_f filesep out.template '*' ext_f], folder_out);
    end
end

if ~isempty(folder_out)
    for ss = 1:length(list_sub)
        f_out = out.flow_field.(list_sub{ss});
        [path_r,name_r,ext_r] = fileparts(spm_in{1}{ss});
        if isempty(path_r)
            path_r = '.';
        end
        movefile(f_out, folder_out);
    end
end
