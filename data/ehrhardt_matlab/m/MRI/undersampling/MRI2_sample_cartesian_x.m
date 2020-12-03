function freq = MRI2_sample_cartesian_x(s_kspace, n_readouts, i_phaseencodes)
% MRI2_sample_cartesian_x
%   freq = MRI2_sample_cartesian_x(s_kspace, n_readouts, i_phaseencodes) samples 
% the kspace in a cartesian manner.
%
% Input:
%   s_kspace [vector]
%       size of the k-space in which we want to sample.
%
%   n_readouts [int]
%       number of readouts in the frequency encoding direction
%
%   i_phaseencodes [vector]
%       indices for the phase encodes
%
% Output:
%   freq [matrix]
%       matrix where each column are the frequencies of a data point in
%       kspace.
%
% See also: MRI3_sample_cartesian_x MRI_sample_line
%
% -------------------------------------------------------------------------
%   changes:
%       2015-06-17: Changed the default behaviour when the number of
%       readouts is not specified but three arguments are given. 
%       It now recognizes this as not specified.
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
    if nargin < 3; i_phaseencodes = 1 : s_kspace(2); end;
        
    freq = zeros(2,n_readouts*length(i_phaseencodes));    
        
    i_start_x = -ceil((s_kspace(1)-1)/2);
    i_end_x = floor((s_kspace(1)-1)/2);
    
    K = n_readouts;    
    i_phaseencodes = i_phaseencodes - 1;
    
    for n = 1 : length(i_phaseencodes)                        
        o = i_phaseencodes(n)-ceil((s_kspace(2)-1)/2);
        i_start = [i_start_x o];
        i_end = [i_end_x o];
        
        freq(:, K*(n-1)+1 : K*n)  = MRI_sample_line(i_start, i_end, n_readouts);
    end
       
end