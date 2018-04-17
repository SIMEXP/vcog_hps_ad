function [in,out,opt] = spm_brick_preproc(in,out,opt)
% [IN,OUT,OPT] = SPM_BRICK_PREPROC(IN,OUT,OPT)
%
% IN (string) a T1 volume

% OUT.GM_PVE (string, default [c1 IN]) the grey matter PVE
% OUT.WM_PVE (string, default [c2 IN]) the white matter PVE
% OUT.CSF_PVE (string, default [c3 IN]) the cerebrospinal fluid PVE
% OUT.GM_PVE_R (string, default [rc1 IN]) the grey matter PVE, but aligned
%   to a common space to use with DARTEL
% OUT.WM_PVE_R (string, default [rc2 IN]) the white matter PVE, but aligned
%   to a common space to use with DARTEL
% OUT.CSF_PVE_R (string, default [rc3 IN]) the white matter PVE, but aligned
%   to a common space to use with DARTEL
% OUT.DEF (string, default ['y_' IN]) the deformation fields
% OUT.INV_DEF (string, default ['iy_' IN]) the inverse deformation fields
% OUT.PROV (string, default [IN '_seg8.mat'] provenance file for the
%   segmentation
%
% OPT.FOLDER_OUT (string, default same as IN) the folder where to generate outputs
% OPT.FLAG_TEST (boolean, default false) flag to run a test without generating outputs.
%
% If OPT.FLAG_TEST is true, the brick does nothing but still updates the
%    default file names in OUT, as well as default options in OPT.
% By default (OUT = struct) the brick generates all output files.
%    If file names are specified in OUT, those will be used over defaults.
%    If a file name 'skip' is specified, the output is not generated.
%
% This is an interface to the spm_preprocess tool from SPM12.
% It performs tissue segmentation on an MRI T1 scan.
%
% _________________________________________________________________________
% Copyright (c) Angela Tam, Yasser Iturria-Medina, Pierre Bellec
% Montreal Neurological Institute, 2017
% Centre de recherche de l'institut de geriatrie de Montreal,
% Department of Computer Science and Operations Research
% University of Montreal, Quebec, Canada, 2017
% Maintainer : pierre.bellec@criugm.qc.ca
% See licensing information in the code.
% Keywords : SPM, segmentation, interface

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
    error('spm:brick','syntax: [IN,OUT,OPT] = SPM_BRICK_PREPROCESS(IN,OUT,OPT).\n Type ''help spm_brick_preprocess'' for more info.')
end

%% Input
if ~ischar(in)
  error('IN should be a string')
end
[path_f,name_f,ext_f] = fileparts(in);
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
        { 'gm_pve' , 'wm_pve' , 'csf_pve' , 'gm_pve_r' , 'wm_pve_r' , 'csf_pve_r' , 'def' , 'inv_def' , 'prov' } , ...
        { ''       , ''       , ''        , ''         , ''         , ''          ,  ''   , ''        , ''     });

%% Options
if nargin < 3
  opt = struct;
end
opt = psom_struct_defaults(opt , ...
        {'folder_out', 'flag_test'} , ...
        {''          , false      });

spm_path = fileparts(which('spm.m'));
opt_preproc.channel.biasreg = 0.001;
opt_preproc.channel.biasfwhm = 60;
opt_preproc.channel.write = [0 0];
opt_preproc.tissue(1).tpm = {[spm_path,'/tpm/TPM.nii,1']};
opt_preproc.tissue(1).ngaus = 1;
opt_preproc.tissue(1).native = [1 1];
opt_preproc.tissue(1).warped = [0 0];
opt_preproc.tissue(2).tpm = {[spm_path,'/tpm/TPM.nii,2']};
opt_preproc.tissue(2).ngaus = 1;
opt_preproc.tissue(2).native = [1 1];
opt_preproc.tissue(2).warped = [0 0];
opt_preproc.tissue(3).tpm = {[spm_path,'/tpm/TPM.nii,3']};
opt_preproc.tissue(3).ngaus = 2;
opt_preproc.tissue(3).native = [1 1];
opt_preproc.tissue(3).warped = [0 0];
opt_preproc.tissue(4).tpm = {[spm_path,'/tpm/TPM.nii,4']};
opt_preproc.tissue(4).ngaus = 3;
opt_preproc.tissue(4).native = [0 0];
opt_preproc.tissue(4).warped = [0 0];
opt_preproc.tissue(5).tpm = {[spm_path,'/tpm/TPM.nii,5']};
opt_preproc.tissue(5).ngaus = 4;
opt_preproc.tissue(5).native = [0 0];
opt_preproc.tissue(5).warped = [0 0];
opt_preproc.tissue(6).tpm = {[spm_path,'/tpm/TPM.nii,6']};
opt_preproc.tissue(6).ngaus = 2;
opt_preproc.tissue(6).native = [0 0];
opt_preproc.tissue(6).warped = [0 0];
opt_preproc.warp.mrf = 1;
opt_preproc.warp.cleanup = 1;
opt_preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
opt_preproc.warp.affreg = 'mni';
opt_preproc.warp.fwhm = 0;
opt_preproc.warp.samp = 3;
opt_preproc.warp.write = [1 1];
opt_preproc.folder_out = '';
opt_preproc.flag_test = opt.flag_test;

opt = psom_defaults(opt_preproc, opt );

if strcmp(opt.folder_out,'')
    opt.folder_out = path_f;
end

%% Building default output names
if ~strcmp(opt.folder_out, path_f)
    ffields = fieldnames(out);
    for ff = 1:length(fieldnames(out))
        ffield = ffields{ff};
        if ~isempty(out.(ffield))
            out.(ffield) = [opt.folder_out filesep out.(ffield)];
        end
    end
end

if isempty(out.gm_pve)
    out.gm_pve = [opt.folder_out filesep 'c1' name_f ext_f];
end
if isempty(out.wm_pve)
    out.wm_pve = [opt.folder_out filesep 'c2' name_f ext_f];
end
if isempty(out.csf_pve)
    out.csf_pve = [opt.folder_out filesep 'c3' name_f ext_f];
end
if isempty(out.gm_pve_r)
    out.gm_pve_r = [opt.folder_out filesep 'rc1' name_f ext_f];
end
if isempty(out.wm_pve_r)
    out.wm_pve_r = [opt.folder_out filesep 'rc2' name_f ext_f];
end
if isempty(out.csf_pve_r)
    out.csf_pve_r = [opt.folder_out filesep 'rc3' name_f ext_f];
end
if isempty(out.def)
    out.def = [opt.folder_out filesep 'y_' name_f ext_f];
end
if isempty(out.inv_def)
    out.inv_def = [opt.folder_out filesep 'iy_' name_f ext_f];
end
if isempty(out.prov)
    out.prov = [opt.folder_out filesep name_f '_seg8.mat'];
end

if opt.flag_test == 1
    return
end

%%%%%%%%%%%%%%%%
%% The brick really starts here
%%%%%%%%%%%%%%%%

%% do the segmentation
folder_out = opt.folder_out;
opt = rmfield ( opt , { 'folder_out' , 'flag_test'}); % Get rid of the options not supported by spm_preproc_run
opt.channel.vols = {in};
spm_preproc_run(opt);

%% move the outputs to opt.folder_out
if ~strcmp(folder_out, path_f)
    movefile([path_f filesep 'c1' name_f ext_f], out.gm_pve)
    movefile([path_f filesep 'c2' name_f ext_f], out.wm_pve)
    movefile([path_f filesep 'c3' name_f ext_f], out.csf_pve)
    movefile([path_f filesep 'rc1' name_f ext_f], out.gm_pve_r)
    movefile([path_f filesep 'rc2' name_f ext_f], out.wm_pve_r)
    movefile([path_f filesep 'rc3' name_f ext_f], out.csf_pve_r)
    movefile([path_f filesep 'y_' name_f ext_f], out.def)
    movefile([path_f filesep 'iy_' name_f ext_f], out.inv_def)
    movefile([path_f filesep  name_f '_seg8.mat'], out.prov)
end



