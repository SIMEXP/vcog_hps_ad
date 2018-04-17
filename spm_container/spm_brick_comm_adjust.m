function [in,out,opt] = spm_brick_comm_adjust(in,out,opt)
% [IN,OUT,OPT] = SPM_BRICK_COMM_ADJUST(IN,OUT,OPT)
%
% IN (string) a T1 volume
% OUT (string, default ['acpc_' IN]) the adjusted volume
% OPT.FOLDER_OUT (string, default same as IN) the folder where to generate outputs
% OPT.FLAG_TEST (boolean, default false) flag to run a test without generating outputs
%
% If OPT.FLAG_TEST is true, the brick does nothing but still updates the
%    default file names in OUT, as well as default options in OPT.
% By default (OUT = struct) the brick generates all output files.
%    If file names are specified in OUT, those will be used over defaults.
%    If a file name 'skip' is specified, the output is not generated.
%
% This is an interface to SPM.
% It reorients an image to the AC-PC line.
%
% _________________________________________________________________________
% Copyright (c) Angela Tam, Yasser Iturria-Medina, Pierre Bellec
% Montreal Neurological Institute, 2017
% Centre de recherche de l'institut de geriatrie de Montreal,
% Department of Computer Science and Operations Research
% University of Montreal, Quebec, Canada, 2017
% Maintainer : pierre.bellec@criugm.qc.ca
% See licensing information in the code.
% Keywords : SPM, interface

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

%% Syntax checks

if ~exist('in','var')||~exist('out','var')||~exist('opt','var')
    error('spm:brick','syntax: [IN,OUT,OPT] = SPM_BRICK_COMM_ADJUST(IN,OUT,OPT).\n Type ''help spm_brick_comm_adjust'' for more info.')
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
  out = '';
end

%% Options
if nargin < 3
  opt = struct;
end
opt = psom_struct_defaults(opt , ...
        { 'folder_out' , 'flag_test' } , ...
        { ''           , false       });

if strcmp(opt.folder_out,'')
    opt.folder_out = path_f;
end

%% Building default output names
if isempty(out)
    out = [opt.folder_out filesep 'acpc_' name_f ext_f]; 
end
if opt.flag_test == 1
    return
end


%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Brick starts here
%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get volume header info
vol = spm_vol(in);

%% Set the voxel-to-world mapping of the volume
vol.mat = diag([sqrt(sum(vol.mat(1:3,1:3).^2)) 1]);
vol.mat(:,4) = [(-1)*sign(diag(vol.mat(1:3,1:3))).*vol.dim(1:3)'/2; 1];
in_a.img = in;
in_a.map = vol.mat;
out_a = struct;
opt_a.folder_out = opt.folder_out;
[~, out_a, ~] = spm_brick_get_space(in_a, out_a, opt_a);

Tmp = sprintf('%s\\templates\\%s.nii',spm('Dir'),'T1');
           
%% 3 dimensional convolution of an image to 8bit data in memory           
VRef = spm_smoothto8bit(spm_vol(in),8);
try
    VTmp = spm_smoothto8bit(spm_vol(Tmp),0);
catch
    Tmp = sprintf('%s\\canonical\\%s.nii',spm('Dir'),['avg152' 'T1']);
    VTmp = spm_smoothto8bit(spm_vol(Tmp),0);
end
VRef.pinfo(1:2,:) = VRef.pinfo(1:2,:)/spm_global(VRef);
VTmp.pinfo(1:2,:) = VTmp.pinfo(1:2,:)/spm_global(VTmp);

%% Affine registration using least squares.
% Options
sep = 8./[1 2 4*ones(1,2)];
flags = struct('WG'      ,[]    ,...
               'WF'      ,[]    ,...
               'sep'     ,8     ,...
               'regtype' ,'mni' ,...
               'globnorm',0);
[M,scal] = spm_affreg(VTmp,VRef,flags,eye(4));

% Second Affine registration using least squares.
flags.sep = sep(2);
[M,scal] = spm_affreg(VTmp,VRef,flags,M,scal); %VTmp.mat\M*VRef.mat

[A,B,C] = svd(M(1:3,1:3));
R = A*C';
R(:,4) = R*(M(1:3,1:3)\M(1:3,4));
R(4,4) = 1;

%% Set the voxel-to-world mapping of the volume again
in_b.img = out_a.vol;
in_b.map = '';
out_b = struct;
opt_b.folder_out = opt.folder_out;
[~, out_b, ~] = spm_brick_get_space(in_b, out_b, opt_b);

in_c.img = out_a.vol;
in_c.map = R*out_b.map;
out_c.vol = out;
opt_c.folder_out = opt.folder_out;
spm_brick_get_space(in_c, out_c,opt_c);


