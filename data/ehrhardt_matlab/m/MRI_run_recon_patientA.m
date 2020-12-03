% MRI_run_recon_patientA
%   This script runs several reconstruction procedures for "patientA"
%
% -------------------------------------------------------------------------
%   changes:
% 
% 2015-06-17 --------------------------------------------------------------
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

name = 'patientA';

load(['data/data_paper/' name '/' name '.mat']);

subfolder = ''; %My_date;

% alphas = [1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1];
% etas = 1e-2;

alphas = [1e-4 1e-3 1e-2 1e-1];
etas = [1e-4 1e-3 1e-2 1e-1 1e-0];

%% select data
array_contrast = {'T1','T2'};

% array_data = fieldnames(data.mri.T1);

% array_data = {...
%     'spiral8' ...
%     'cartesianX_random_0_32' ...
%     'cartesianY4' ...
%     };

array_data = {...
    'cartesianY4' ...
    };

%% run recon
run_without_prior = true;
run_TV = true;
run_wTV = true;
run_dTV = true;
MRI_run_recon;