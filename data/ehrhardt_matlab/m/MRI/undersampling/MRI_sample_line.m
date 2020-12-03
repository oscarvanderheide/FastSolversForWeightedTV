function frequencies = MRI_sample_line(k_start, k_end, n_samples)
% MRI_sample_line
%   frequencies = MRI_sample_line(k_start, k_end, n_samples) samples the 
% k-space along an affine trajectory.
%
% Input:
%   k_start [vector]
%       the n-dimensional vector specifies the START location of the
%       trajectory.
%
%   k_end [vector]
%       the n-dimensional vector specifies the FINAL location of the
%       trajectory.
%
%   n_samples [int]
%       the number of samples along the trajectory
%
% Output:
%   frequencies [matrix]
%       output matrix of size n x n_samples. Each column specifies the
%       coordinates of a sample point in k-space.
%
% See also:
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

    dim = length(k_start);
    frequencies = zeros(dim, n_samples);
    
    for n = 1 : dim
        frequencies(n,:) = linspace(k_start(n), k_end(n), n_samples);
    end
    
end
