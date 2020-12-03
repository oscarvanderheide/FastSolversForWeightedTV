function x = Uifftn(X)
% Uifftn
%   x = Uifftn(X) computes the unitary and centered n-D inverse Fourier transform.
%
% Input:
%   X [matrix]
%       n-D k-space
%
% Output:
%   x [matrix]
%       n-D image
%
% See also: Ufftn
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
    
    x = fftshift(ifftn(ifftshift(X))) * sqrt(numel(X));
    
end