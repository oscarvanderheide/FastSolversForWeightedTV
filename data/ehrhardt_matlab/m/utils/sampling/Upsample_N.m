function X = Upsample_N(x, N)
% Upsample_N
%   X = Upsample_N(x, N) samples an input image up by a factor of N with
% constant values in the areas. Its scaled to be the adjoint of
% Downsample_N.
%
% Input:    
%   x [matrix]              
%       scalar valued function
%
% Output:
%   X [matrix]
%       upscaled output
%
% See also: Upsample_2 Downsample_2 Downsample_N
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

    N = round(log2(N));
    
    for n = 1 : N
        x = Upsample_2(x);
    end
    
    X = x;
    
end