function [in,out,opt] = spm_brick_get_space(in,out,opt)
% [IN,OUT,OPT] = SPM_BRICK_GET_SPACE(IN,OUT,OPT)
%
% Get and/or set the voxel-to-world mapping of an image
%
% IN (structure, with the followings fields):
%   IMG (string, optional) filename of an image
%   MAP (double, optional) a 4 x 4 matrix of the voxel-to-world mapping
% OUT (structure, with the following fields):
%   MAP (optional, double) a 4 x 4 matrix of the voxel-to-world mapping from IN.IMG
%   VOL (optional, string, default ['adj_' in.img]) filename of the resulting adjusted image
% OPT (structure, with the following fields):
%   FOLDER_OUT (string, default same as IN) the folder where to generate outputs
%   FLAG_TEST (boolean, default false) flag to run a test without generating outputs
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
    error('spm:brick','syntax: [IN,OUT,OPT] = SPM_BRICK_GET_SPACE(IN,OUT,OPT).\n Type ''help spm_brick_get_space'' for more info.')
end

%% Input
if ~isstruct(in)
  error('IN should be a structure')
end
[path_f,name_f,ext_f] = fileparts(in.img);
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
        { 'vol' , 'map'} , ...
        { ''    , ''   });

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
if isempty(out.vol)
    out.vol = [opt.folder_out filesep 'adj_' name_f ext_f];  %%%%% need to figure out 
end
if opt.flag_test == 1
    return
end

%%%%%%%%%%%%%%%
%% Brick starts here
%%%%%%%%%%%%%%%

[~,~,~,num] = spm_fileparts(in.img);
if ~isempty(num)
    n = str2num(num(2:end));
else
    n = [1 1];
end

% Get header info and read the volume
hdr = spm_vol(fullfile(in.img));
[vol, ~] = spm_read_vols(hdr);

if ~isempty(in.img) && ~isempty(in.map) %nargin==2
    %% Create a new volume with new voxel-to-world mapping
    if isempty(hdr.private.mat_intent) 
        hdr.mat = hdr.mat; 
        hdr.private.mat = hdr.private.mat;
    end
    hdr.mat_intent = 'Aligned';
    if n(1)==1

        % Ensure volumes 2..N have the original matrix
        if size(hdr.private.dat,4)>1 && sum(sum((hdr.private.mat-in.map).^2))>1e-8
            M0 = hdr.private.mat;
            if ~isfield(hdr.private.extras,'mat')
                hdr.private.extras.mat = zeros([4 4 size(hdr.private.dat,4)]);
            else
                if size(hdr.private.extras.mat,3)<size(hdr.private.dat,4)
                    hdr.private.extras.mat(:,:,size(hdr.private.dat,4)) = zeros(4);
                end
            end
            for i=2:size(hdr.private.dat,4)
                if sum(sum(hdr.private.extras.mat(:,:,i).^2))==0
                    hdr.private.extras.mat(:,:,i) = M0;
                end
            end
        end

        hdr.mat = in.map;
        hdr.private.mat = in.map;
        if strcmp(hdr.private.mat0_intent,'Aligned') 
            hdr.private.mat0 = in.map; 
        end
        if ~isempty(hdr.private.extras) && isstruct(hdr.private.extras) && isfield(hdr.private.extras,'mat') &&...
            size(hdr.private.extras.mat,3)>=1
            hdr.private.extras.mat(:,:,n(1)) = in.map;
        end
     else
         hdr.private.extras.mat(:,:,n(1)) = in.map;
    end
    % write new volume
    hdr.fname = out.vol; 
    spm_write_vol(hdr,vol);
else
    %% get the space map from input volume
    out.vol = 'no volume outputted';
    if ~isempty(hdr.private.extras) && isstruct(hdr.private.extras) && isfield(hdr.private.extras,'mat') &&...
            size(hdr.private.extras.mat,3)>=n(1) && sum(sum(hdr.private.extras.mat(:,:,n(1)).^2))
        out.map = hdr.private.extras.mat(:,:,n(1));
    else
        out.map = hdr.private.mat;
    end
end
