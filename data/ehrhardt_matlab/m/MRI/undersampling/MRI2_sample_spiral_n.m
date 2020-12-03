function freq = MRI2_sample_spiral_n(s_kspace, n_readouts, n_turns, uniformity, exponent, first_angles)
% MRI2_sample_spiral_n
%   freq = MRI2_sample_spiral(s_kspace, n_readouts, n_turns, uniformity, exponent, first_angles) 
% samples the kspace spirally with n spirals.
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
%   uniformity [scalar; DEFAULT = see MRI2_sample_spiral] 
%       constant for spiral. If uniformity = 1, the spiral is homogenuous.
%       In case uniformity < 1 it is widening, and if uniformity > 1 it is 
%       denser in the center.
%
%   exponent [scalar; DEFAULT = see MRI2_sample_spiral]
%       constant for spiral can be used to change the distance between 
%       adjacent samples.
%
%   first_angles [vector / int; DFAULT = 3]
%       how many spirals do you want? If first_angles is an integer, the
%       spirals have uniform first angles.
%
% Output:
%   freq [matrix]
%       matrix where each column are the frequencies of a data point in
%       kspace.
%
% See also: MRI2_sample_spiral
%
% -------------------------------------------------------------------------
%   changes:
% 
% 2015-06-19 --------------------------------------------------------------
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

    if nargin < 4; uniformity = []; end;
    if nargin < 5; exponent = []; end;
    if nargin < 6; first_angles = 3; end;
    
    if numel(first_angles) == 1; first_angles = linspace(0,2*pi*(first_angles-1)./first_angles,first_angles); end;
    
    freq = [];
    for i = 1 : length(first_angles)
        freq = [freq MRI2_sample_spiral(s_kspace, n_readouts, n_turns, ...
            uniformity, exponent, first_angles(i))];
    end
    
end