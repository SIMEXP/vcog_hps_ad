function [in, out, opt] = spm_brick_dartel_norm(in,out,opt)

% This function will warp individual structural images to match a template
%
% IN (structure) with the following fields:
%   IMAGES.<SUBJECT> (string) filename for image to be registered
%   FLOWFIELDS.<SUBJECT> (string) filename for a flowfield image
%   NOTE: The subject order between IMAGES.<SUBJECT> and
%   FLOWFIELDS.<SUBJECT> must be the same
%
% OUT (structure) with the following fields:
%   GM_WARPED.<SUBJECT> (string, default ['mw' IMAGES.<SUBJECT>) filename 
%       of the resulting grey matter images warped to template space
%   SMOOTH.<SUBJECT> (string, default ['smw' IMAGES.<SUBJECT>) filename of 
%       the resulting smoothed warped grey matter images
%
% OPT.FOLDER_OUT (string, default same as IN.FLOWFIELDS.<SUBJECT>) the folder 
%       where to generate outputs. If left empty, the outputs for a given
%       image in IN.FLOWFIELDS.<SUBJECT> will be written to the same
%       originating path (i.e. path of IN.FLOWFIELDS.<SUBJECT> == path of 
%       outputs from IN.FLOWFIELDS.<SUBJECT>).
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
    error('spm:brick','syntax: [IN,OUT,OPT] = SPM_BRICK_DARTEL_NORM(IN,OUT,OPT).\n Type ''help spm_brick_dartel_norm'' for more info.')
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
if nargin < 2
  out = struct;
end
out = psom_struct_defaults(out , ...
        { 'gm_warped' , 'smooth'   } , ...
        { struct      , struct     });

%% Options
if nargin < 3
  opt = struct;
end
opt = psom_struct_defaults(opt , ...
        { 'flag_test', 'folder_out' },...
        { false      , ''           });
    
%% Building default output names
list_sub = fieldnames(in.images);
for ss = 1:length(list_sub)
    [path_f,name_f,ext_f] = fileparts(in.images.(list_sub{ss}));
    if isempty(path_f)
        path_f = '.';
    end
    if isempty(out.gm_warped) || isempty(fieldnames(out.gm_warped))
        out.gm_warped.(list_sub{ss}) = [path_f filesep 'mw' name_f ext_f];
    end
    if isempty(out.smooth) || isempty(fieldnames(out.smooth))
        out.smooth.(list_sub{ss}) = [path_f filesep 'smw' name_f ext_f];
    end
end

if opt.flag_test == 1
    return
end

%% Brick really starts here
spm_img = struct2cell(in.images);
spm_ff = struct2cell(in.flowfields);

%% warp to group-specific template space
w_opt = struct;
w_opt.images = {spm_img};
w_opt.jactransf = 1;
w_opt.K = 6;
w_opt.interp = 7;
w_opt.flowfields = spm_ff;
spm_dartel_norm(w_opt);

%% smooth the warped images
s_opt = struct;
s_opt.fwhm = [8 8 8];
s_opt.dtype = 0;
for ff = 1:length(list_sub)
    [path_r,~,~] = fileparts(spm_ff{ff});
    [~,name_r,ext_r] = fileparts(spm_img{ff});
    if isempty(path_r)
        path_r = '.';
    end
    s_in = [path_r filesep 'mw' name_r ext_r];
    s_out = [path_r filesep 'smw' name_r ext_r];
    spm_smooth(s_in, s_out, s_opt.fwhm, s_opt.dtype);
end

%% rename the warped images outputs to user-specified filenames
for ss = 1:length(list_sub)
    mw_out = out.gm_warped.(list_sub{ss});
    s_out = out.smooth.(list_sub{ss});
    [path_r,~,~] = fileparts(spm_ff{ff});
    [~,name_r,ext_r] = fileparts(spm_img{ff});
    if isempty(path_r)
        path_r = '.';
    end
    mw_file = [path_r filesep 'mw' name_r ext_r];
    s_file = [path_r filesep 'smw' name_r ext_r];
    if ~strcmp(mw_out, mw_file)
        movefile(mw_file, mw_out);
    end
    if ~strcmp(s_out, s_file)
        movefile(s_file, s_out);
    end
end

%% move the outputs to opt.folder_out if specified
if ~isempty(opt.folder_out)
    for ss = 1:length(list_sub)
        [path_r,~,~] = fileparts(spm_ff{ff});
        mw_out = out.gm_warped.(list_sub{ss});
        s_out = out.smooth.(list_sub{ss});
        [~,name_r,ext_r] = fileparts(spm_ff{ff});
        if isempty(path_r)
            path_r = '.';
        end
        movefile([path_r filesep mw_out], opt.folder_out);
        movefile([path_r filesep s_out], opt.folder_out);
    end
end