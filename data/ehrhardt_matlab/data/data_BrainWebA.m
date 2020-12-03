% data_BrainWebA.m
%   data_BrainWebA creates the data set "data_BrainWebA" which is a transverse slice.
%
% See also: 
%
% 2015-10-14 --------------------------------------------------------------
% Matthias J. Ehrhardt
% CMIC, University College London, UK 
% matthias.ehrhardt.11@ucl.ac.uk
% http://www.cs.ucl.ac.uk/staff/ehrhardt/software.html
%
% -------------------------------------------------------------------------
% Copyright 2015 University College London
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%   http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.
% -------------------------------------------------------------------------

options.misc.name = 'BrainWebA';

options.misc.folder.data = ['results/data/' options.misc.name];
options.misc.folder.results = ['results/' options.misc.name];

mkdir(options.misc.folder.data);
mkdir(options.misc.folder.results);

%% Load data set
load('data_BrainWebA');
options.mri.resolution = [1 1]; % in mm, assuming isotropic pixels
options.mri.objectsize_in_pixel = size(groundtruth.T1);

D = fieldnames(groundtruth);
for d = 1 : length(D)
    eval(sprintf('x = groundtruth.%s;',D{d}))
    C = 256;
    imwrite((C-1)*x,colormap(gray(C)),sprintf('%s/%s_groundtruth_%s.png', options.misc.folder.data, options.misc.name, D{d}));
end

%%
mask = groundtruth.T2 > .3;
mask = imclose(mask,fspecial('disk', 13)>1e-16);
clf; imagesc(groundtruth.T1 + mask); axis image; colorbar;
options.mri.roi = int64(find(mask));
saveas(gcf, [options.misc.folder.data '/' options.misc.name '_mask.png']);

%% kspace
FT1 = Ufftn(groundtruth.T1);
FT2 = Ufftn(groundtruth.T2);

%% sampling
set_sampling