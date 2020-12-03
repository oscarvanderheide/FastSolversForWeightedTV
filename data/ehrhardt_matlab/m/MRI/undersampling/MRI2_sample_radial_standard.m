function freq = MRI2_sample_radial_standard(s_kspace, n_readouts, n_spokes, offset_angle)
% MRI2_sample_radial_standard
%   freq = MRI2_sample_radial_standard(s_kspace, n_readouts, n_angles, offset_angle) 
% samples the kspace radially with restricted parameters.
%
% Input:
%   s_kspace [vector; DEFAULT = [128 128]]
%       size of the k-space in which we want to sample.
%
%   n_readouts [int]
%       number of readouts in the frequency encoding direction
%
%   n_spokes [int; DEFAULT = 10]
%       number of spokes
%
%   offset_angle [scalar; DEFAULT = 0]
%       dont start at angle 0 but at angle "offset_angle"
%
% Output:
%   freq [matrix]
%       matrix where each column are the frequencies of a data point in
%       kspace.
%
% See also: MRI2_sample_radial MRI_sample_line
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

    if nargin < 1 || numel(s_kspace) == 0; s_kspace = [128 128]; end;
%     if nargin < 2 || numel(n_readouts) == 0; n_readouts = s_kspace(1)-1; end;
    if nargin < 3 || numel(n_spokes) == 0; n_spokes = 10; end;
    if nargin < 4 || numel(offset_angle) == 0; offset_angle = 0; end;
    
    i_angles = pi*linspace(0, (n_spokes-1)/n_spokes, n_spokes) + offset_angle;
        
    freq = MRI2_sample_radial(s_kspace, n_readouts, i_angles);
    
end