function c = colour2colour(c1,c2,N)
% colour2colour
%   c = colour2colour(c1,c2,N) returns a colourmap with N colours that is a
% linear interpolation between the colours c1 and c2.
%
% Input:
%   c1 [1x3 matrix]
%       specifies colour 1 in RGB values
%
%   c2 [1x3 matrix]
%       specifies colour 2 in RGB values
%
%   N [int]
%       number of colours in the colourmap
%
% Output:
%   c [Nx3 matrix]
%       colourmap from c1 to c2
%        
% See also:
%
% -------------------------------------------------------------------------
%   changes:
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

    c = zeros(N,3);
    
    for k = 1 : 3
        c(:,k) = linspace(c1(k),c2(k),N);
    end
    
end