function fun = Penalty_dTV(options, u, v, grad)
% Penalty_dTV
%   [fun, grad, fun_image] = Penalty_dTV(options, u, v) computes the dTV 
% semi norm of an input image u with a given normalized vector field v. 
%
% Input:
%   options [struct]
%       a struct with the following fields
%           options.alpha: an overall scaling parameter
%           options.s_voxel: the size of the voxel in arbitrary units.
%           options.s_image: the size of the image in voxels.
%       
%   u [matrix] 
%       2D image
%       
%   v [matrix] 
%       2D normalized vector field
%
% Output:
%   fun [scalar]
%       function value of the prior
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

    u = reshape(u, options.s_image);
    
    % calulate the gradient
    Gu = Gradient2D_forward_constant_unitstep(u, grad);
    
    % project it on vector field 
    PGu = Vec2D_proj_orth(Gu,v);
    
    % and store its norm
    NPGu = Norm_n(PGu, 0, 3);
    
    fun_image = options.alpha * NPGu;
    fun = sum(fun_image(:));
      
end