function freq = MRI2_sample_radial_random(s_kspace, n_readouts, n_spokes)
% MRI2_sample_radial_random
%   freq = MRI2_sample_radial_random(s_kspace, n_readouts, n_spokes) 
% samples the kspace radially and the spokes are uniformly distributed.
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
    if nargin < 3 || numel(n_spokes) == 0; n_spokes = 10; end;
    
    JJ = randperm(360*10);
    JJ = JJ(1:n_spokes)/(360*10);    
    i_angles = pi*(n_spokes-1)/n_spokes*JJ;
        
    freq = MRI2_sample_radial(s_kspace, n_readouts, i_angles);
    
end