function freq = MRI2_sample_cartesian_y_standard(s_kspace, n_readouts, phase_jump, phase_offset)
% MRI2_sample_cartesian_y_standard
%   freq = MRI2_sample_cartesian_y_standard(s_kspace, n_readouts, i_phaseencodes) samples 
% the kspace in a cartesian manner with equidistant spacing along the
% y-axis.
%
% Input:
%   s_kspace [vector]
%       size of the k-space in which we want to sample.
%
%   n_readouts [int]
%       number of readouts in the frequency encoding direction
%
%   phase_jump [int]
%       equidistant undersampling
%
%   phase_offset [int]
%       offset in the phase encoding direction
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
    if nargin < 2 || numel(n_readouts) == 0; n_readouts = s_kspace(1); end;
    if nargin < 3; phase_jump = 1; end;
    if nargin < 4; phase_offset = 0; end;
        
    freq = MRI2_sample_cartesian_x(s_kspace, n_readouts, (1+phase_offset) : phase_jump : s_kspace(2));
    freq = flipud(freq);
    
end