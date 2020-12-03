function v = Vec2D_mult(v,s)
% Vec2D_mult
%   v = Vec2D_mult(v, s) multiplies a vector field with a scalar field point wise.
%
% Input:
%   v [matrix]
%       input vector field
%
%   s [matrix]
%       input scalar field
%
% Output:
%   v [matrix]
%       normalized vector field
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

    v(:,:,1) = v(:,:,1).*s;
    v(:,:,2) = v(:,:,2).*s;
    
end