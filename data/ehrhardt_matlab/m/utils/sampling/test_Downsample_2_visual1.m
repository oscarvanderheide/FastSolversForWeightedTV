function ok = test_Downsample_2_visual1
% test_Downsample_2_1
%   ok = test_Downsample_2_1 is a unit test for Downsample_2.
%
% Output:
%   ok [boolean]
%       test successful?
%
% See also: Downsample_2
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

    x = phantom3d(64);
    
    xx = Downsample_2(x);
    
    figure(1); imagesc3(x);
    figure(2); imagesc3(xx);
    
    ok = User_decides;
    
end