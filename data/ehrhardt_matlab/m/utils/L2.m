function r = L2(x, x0, ROI)
% L2
%   r = L2(x) computes the l2 norm of the input x. The input is allowed to
% be any kind of vector, matrix, tensor etc.
%
% Input:
%   x [vector]
%       input image
%
% Output:
%   r [scalar]
%       relative l2 norm
%
% See also: Rel_l2 Rel_l2_ROI
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

    switch nargin
        case 1
            r = norm(x(:));
            
        case 2
            r = L2(x-x0);
            
        case 3
            r = L2(x(ROI), x0(ROI));
            
    end
end
    