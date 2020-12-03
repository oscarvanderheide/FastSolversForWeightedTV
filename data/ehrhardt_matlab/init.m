% init.m
%   init adds the necessary paths to run the code.
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

display('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>');
display(' initialize paths');

this_location = fileparts(which('init'));

addpath(this_location);
addpath([this_location, '/data']);
addpath([this_location, '/m']);
addpath([this_location, '/m/methods']);
addpath([this_location, '/m/methods/priors']);
addpath([this_location, '/m/methods/proximal']);
addpath([this_location, '/m/MRI']);
addpath([this_location, '/m/MRI/undersampling']);
addpath([this_location, '/m/utils']);
addpath([this_location, '/m/utils/colourbar']);
addpath([this_location, '/m/utils/derivative']);
addpath([this_location, '/m/utils/norms']);
addpath([this_location, '/m/utils/parameter_estimation']);
addpath([this_location, '/m/utils/stats']);
addpath([this_location, '/m/utils/vector_fields']);

clear this_location;

display('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<');
