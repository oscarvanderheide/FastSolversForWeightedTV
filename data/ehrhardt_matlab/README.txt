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

This folder contains matlab scripts (.m-files) and matlab data (.mat-files) that
have been used to create the results in 

  [1] Matthias J. Ehrhardt, Marta M. Betcke, 
      Multi-Contrast MRI Reconstruction with Structure-Guided Total Variation
      Submitted to SIAM Journal on Imaging Sciences

To initialize these files please run
    init.m.
It adds all the necessary paths to your search path.

There are a few scripts that can be readily run by the user. The user can either run a simple example 
    MRI_run_recon_simple.m
of the four reconstructions with one set of parameters (not necessarily the optimal ones) on one data 
set with one sampling. This takes about 1 min. 

Or the user can do the reconstructions on all six data sets
with a range of parameters thus reproducing the results of the paper by
    MRI_run_recon_all_datasets.m,
Please be aware that the execution of this file might take several hours. The file itself consists of
reconstructions of the single data sets
    MRI_run_recon_phantom.m
    MRI_run_recon_BrainWebA.m
    MRI_run_recon_BrainWebB.m
    MRI_run_recon_BrainWebC.m
    MRI_run_recon_patientA.m
    MRI_run_recon_patientB.m

If you want to modify the code (e.g. changing the regularization parameters) or apply this 
code to your own data set, please check out MRI_run_recon_simple.m.
Please be aware of that the actual .m-files that run the reconstruction need the data in a 
special structure which you need to mimic ro run these files on your data sets.

The code has been successfully tested on
    - OS: Ubuntu 14.04 LTS (64-bit) with MATLAB R2015a
      Processor: Intel(R) Xeon(R) CPU E5-2620 0 @ 2.00 GHz x 18
      Memory: 15.6 GB
    - OS: Windows 8.1 Pro (64-bit) with MATLAB R2015a
      Processor: Intel(R) Core(TM) i5-3320M CPU @ 2.60 GHz 2.60 GHz
      Memory: 8.00 GB (7.70 GB usable)
