function freq = MRI2_sample_cartesian_y_random_quadratic(s_kspace, n_readouts, n_fixed_lines, n_random_lines)
% MRI2_sample_cartesian_y_random_quadratic
%   freq = MRI2_sample_cartesian_y_random_quadratic(s_kspace, n_readouts, n_fixed_lines, n_random_lines) samples 
% the kspace in a cartesian manner variable density in the phase encode
% direction.
%
% Input:
%   s_kspace [vector; DEFAULT = [128 128]]
%       size of the k-space in which we want to sample.
%
%   n_readouts [int; DEFAULT = s_kspace(1)]
%       number of readouts in the frequency encoding direction
%
%   n_fixed_lines [int; DEFAULT = 11]
%       number of fixed, centered lines in the phase encode direction
%
%   n_random_lines [int; DEFAULT = 10]
%       number of random lines in the phase encode direction
%
% Output:
%   freq [matrix]
%       matrix where each column are the frequencies of a data point in
%       kspace.
%
% See also: MRI2_sample_cartesian_x MRI_sample_line
%
% -------------------------------------------------------------------------
%   changes:
% 
% 2014-08-06 --------------------------------------------------------------
% Matthias J. Ehrhardt
% CMIC, University College London, UK 
% matthias.ehrhardt.11@ucl.ac.uk
% http://www.cs.ucl.ac.uk/staff/ehrhardt/software.html
%
% -------------------------------------------------------------------------
% Copyright 2014 University College London
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

    if nargin < 1; s_kspace = [128 128]; end;
    if nargin < 2; n_readouts = s_kspace(1); end;
    if nargin < 3; n_fixed_lines = 11; end;
    if nargin < 4; n_random_lines = 10; end;
    
    freq = MRI2_sample_cartesian_x_random_quadratic(s_kspace, n_readouts, n_fixed_lines, n_random_lines);
    freq = flipud(freq);
       
end