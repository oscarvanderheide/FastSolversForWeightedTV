% MRI_run_recon_all_data_sets
%   This script runs several reconstruction procedures for all six phantoms
%
% -------------------------------------------------------------------------
%   changes:
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

clear all;
clf;

display('This code takes about 2h to run! To proceed press: enter, to cancel press: Ctrl+C'); pause();

time = [];

t = tic; MRI_run_recon_phantom; time = [time toc(t)];

t = tic; MRI_run_recon_BrainWebA; time = [time toc(t)];
t = tic; MRI_run_recon_BrainWebB; time = [time toc(t)];
t = tic; MRI_run_recon_BrainWebC; time = [time toc(t)];

t = tic; MRI_run_recon_patientA; time = [time toc(t)];
t = tic; MRI_run_recon_patientB; time = [time toc(t)];

sum(time)/60