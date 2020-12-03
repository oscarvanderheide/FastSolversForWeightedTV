function X = Ufftn(x)
% Ufftn
%   x = Ufftn(X) computes the unitary and centered n-D Fourier transform.
%
% Input:
%   x [matrix]
%       n-D image
%
% Output:
%   X [matrix]
%       n-D k-space
%
% See also: Uifftn
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
    
    X = fftshift(fftn(ifftshift(x))) / sqrt(numel(x));
    
end