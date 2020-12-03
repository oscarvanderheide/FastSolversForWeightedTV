function r = Rel_l2(x, x0, ROI)
% Rel_l2
%   r = Rel_l2(x, x0, ROI) computes the relative l2 error of x with respect 
% to x0. If a region of interest is defined, it restricts the error to this 
% region.
%
% Input:
%   x [vector]
%       input image
%
%   x0 [vector]
%       reference image
%
%   ROI [vector]
%       list of indices defining the region of interest
%
% Output:
%   r [scalar]
%       relative l2 error
%
% See also: L2 Rel_l2_ROI
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

    if nargin < 3
        % compute relative error with respect to x0
        r = L2(x - x0) / (L2(x0) + 1e-12);
    else
        r = Rel_l2_ROI(x, x0, ROI);
    end
    
end    