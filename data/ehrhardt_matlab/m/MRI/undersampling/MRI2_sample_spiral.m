function freq = MRI2_sample_spiral(s_kspace, n_readouts, n_turns, uniformity, exponent, first_angle)
% MRI2_sample_spiral
%   freq = MRI2_sample_spiral(s_kspace, n_readouts, n_turns, uniformity, exponent, first_angle) 
% samples the kspace spirally.
%
% Input:
%   s_kspace [vector]
%       size of the k-space in which we want to sample.
%
%   n_readouts [int]
%       number of readouts in the frequency encoding direction
%
%   n_turns [scalar]
%       number of turns of the spiral
%
%   uniformity [scalar; DEFAULT = 1] 
%       constant for spiral. If uniformity = 1, the spiral is homogenuous.
%       In case uniformity < 1 it is widening, and if uniformity > 1 it is 
%       denser in the center. 
%
%   exponent [scalar; DEFAULT = uniformity]
%       constant for spiral can be used to change the distance between 
%       adjacent samples. 
%
%   first_angle [scalar; DEFAULT = 0]
%       constant for spiral. What does it do? 
%
% Output:
%   freq [matrix]
%       matrix where each column are the frequencies of a data point in
%       kspace.
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

    if nargin < 4 || isempty(uniformity); uniformity = 1; end;
    if nargin < 5 || isempty(exponent); exponent = uniformity; end;
    if nargin < 6; first_angle = 0; end;
    
    freq = zeros(2, n_readouts);
    
    R = floor((min(s_kspace)-1)/2);
            
    e = exponent;
    u = uniformity;
    n = n_readouts;
    N = n_turns;
   
    t = linspace(0,1,n);
    p = 2*pi*N * t.^(1./(e+1));
    r = R * t.^(u./(e+1));   
    
    freq(1, :) = r .* cos(p + first_angle);
    freq(2, :) = r .* sin(p + first_angle);
    
end