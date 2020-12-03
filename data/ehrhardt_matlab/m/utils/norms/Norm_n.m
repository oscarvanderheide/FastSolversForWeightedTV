function norm_n = Norm_n(u, beta, dim)
% Norm_n
%   norm_n = Norm_n(u, beta, dim) computes the norm of u at every pixel. 
% If beta > 0 it computes the smoothed norm |x|_beta := sqrt(|x|^2 +
% beta^2) instead. In case the variable dim is provided, then the norm is
% taken of the dim th component. E.g. If u represents the gradient of a
% 3d function, u is 4d and the last dimension specifies the direction of 
% the derivatives, then we might want to compute the norm of these
% gradients. We therefore have to specify dim = 4.
%
% Input:
%   u [matrix]
%       input image
%       
%   beta [scalar; DEFAULT = 0] 
%       smoothing factor
%       
%   dim [int; options but no DEFAULT value] 
%       dimension which shall be used for the norm, see the text above.
%
% Output:
%   norm_n [matrix]
%       the output is of the same size as u. If dim is provided then one
%       dimension is reduced to 1.
%
% See also:
%
% -------------------------------------------------------------------------
%   changes:
% 
% 2014-10-02 --------------------------------------------------------------
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

    if ~exist('beta', 'var'); beta = 0; end;
    
    if ~exist('dim', 'var');
        norm_n = sqrt(abs(u).^2 + beta^2);
    else
        norm_n = sqrt(sum(abs(u).^2, dim) + beta^2);
    end
    
end