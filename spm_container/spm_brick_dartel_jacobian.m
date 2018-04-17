function [in, out, opt] = spm_brick_dartel_jacobian(in,out,opt)
% This function generates Jacobian determinant fields
%
% IN (structure) with the following fields:
%   <SUBJECT> (string) flowfield image
%
% OUT (structure, optional) with the following fields:
%   <SUBJECT> (string) filename of Jacbobian determinant field output
%
% OPT.FOLDER_OUT (string, default same as IN.<SUBJECT>) the folder where 
%       to generate outputs. If left empty, the outputs for a given image
%       in IN.<SUBJECT> will be written to the same originating path (i.e.
%       path of IN.<SUBJECT> == path of outputs from IN.<SUBJECT>).
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
    error('spm:brick','syntax: [IN,OUT,OPT] = SPM_BRICK_DARTEL_JACOBIAN(IN,OUT,OPT).\n Type ''help spm_brick_dartel_jacobian'' for more info.')
end

%% Input
if ~isstruct(in)
  error('IN should be a structure')
end

if ~isstruct(out)
    error('OUT should be a structure')
end

%% Temporary folder
if ~isempty(gb_psom_name_job)
    path_tmp = [tempdir filesep gb_psom_name_job filesep];
else
    path_tmp = tempdir;
end

%% Options
if nargin < 3
  opt = struct;
end
opt = psom_struct_defaults(opt , ...
        { 'flag_test', 'folder_out' },...
        { false      , ''           });

%% settings for DARTEL brick
opt_dartel.K = 6;
opt_dartel.folder_out = '';
opt_dartel.flag_test = opt.flag_test;

opt = psom_defaults(opt_dartel, opt );

%% Building default output names
list_sub = fieldnames(in);
if isempty(out) || isempty(fieldnames(out))
    for ss = 1:length(list_sub)
        [path_f,name_f,ext_f] = fileparts(in.(list_sub{ss}));
        if isempty(path_f)
            path_f = '.';
        end
        out.(list_sub{ss}) = [path_f filesep 'jac_' name_f(3:end) ext_f];
    end
end
    

if opt.flag_test == 1
    return
end

folder_out = opt.folder_out;
opt = rmfield ( opt , { 'folder_out' , 'flag_test'}); % Get rid of the options not supported by spm dartel functions

%% Brick really starts here
spm_in = struct2cell(in);

% saving the determinants of the jacobians
opt.flowfields = spm_in;
spm_dartel_jacobian(opt);

%% rename the jacobian outputs to user-specified filenames
for ss = 1:length(list_sub)
    f_out = out.(list_sub{ss});
    [path_r,name_r,ext_r] = fileparts(spm_in{ss});
    if isempty(path_r)
        path_r = '.';
    end
    jac = [path_r filesep 'jac_' name_r(3:end) ext_r];
    if ~strcmp(f_out, jac)
        movefile(jac, f_out);
    end
end

%% move the outputs to opt.folder_out if specified
if ~isempty(folder_out)
    for ss = 1:length(list_sub)
        f_out = out.(list_sub{ss});
        [path_r,name_r,ext_r] = fileparts(spm_in{ss});
        if isempty(path_r)
            path_r = '.';
        end
        movefile([path_r filesep f_out], folder_out);
    end
end
