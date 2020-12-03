function c = redwhiteblue(N, middle)
% redwhiteblue
%   c = redwhiteblue(N, middle) creates a colourmap from red to white to blue.
%
% Input:    
%   N [int]              
%       number of colours in the colourmap
%
%   middle [int]       
%       which colour is the center, i.e. "white"
%
% Output:
%   c [Nx3 matrix]
%       colourmap
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

    if nargin < 1; N = 64; end;    
    if nargin < 2; middle = N/2; end;
       
    if middle == 0
        c = colour2colour(white,blue,N);
    else
        if middle == N
            c = colour2colour(red,white,N);
        else
            tight = 0;
            N1 = floor(middle+tight);
            N2 = ceil(N-middle);
            c = merge_colours(tight,colour2colour(red,white,N1),colour2colour(white,blue,N2));
        end
    end
    
end