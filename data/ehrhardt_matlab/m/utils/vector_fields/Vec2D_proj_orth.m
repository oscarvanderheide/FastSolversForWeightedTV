function v = Vec2D_proj_orth(v, w)
% Vec2D_proj_orth
%   pv = Vec2D_proj_orth(v, w) projects a 2D vector field v onto the orthogonal
% complement of another 2D vector field w.
%
% Input:
%   v [matrix]
%       input vector field
%       
%   w [matrix]
%       vector field for projection
%
% Output:
%   v [matrix]
%       output vector field
%
% See also:
%
% -------------------------------------------------------------------------
%   changes:
% 
% 2015-05-30 --------------------------------------------------------------
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

    c = Inner_prod_n(v,w,3);
    v(:,:,1) = v(:,:,1) - c .* w(:,:,1);
    v(:,:,2) = v(:,:,2) - c .* w(:,:,2);

end