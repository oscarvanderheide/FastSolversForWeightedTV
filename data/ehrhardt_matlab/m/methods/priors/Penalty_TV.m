function fun = Penalty_TV(options, u, grad)
% Penalty_TV
%   fun = Penalty_TV(options, u, grad) computes the TV semi norm of an 
% input image.
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
% Output:
%   fun [scalar]
%       function value of the prior
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

    if nargin < 3
        grad = zeros([optons.s_image 2]);
    end
    
    u = reshape(u, options.s_image);
    
    % calulate the gradient
    G = Gradient2D_forward_constant_unitstep(u, grad);
    
    % and store its norm
    N = Norm_n(G, 0, 3);
    
    fun_image = options.alpha * N;
    fun = sum(fun_image(:));
    
end