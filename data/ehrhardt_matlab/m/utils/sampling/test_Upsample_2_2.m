function ok = test_Upsample_2_2
% test_Upsample_2_2
%   ok = test_Upsample_2_2 is a unit test for Upsample_2.
%
% Output:
%   ok [boolean]
%       test successful?
%
% See also: Upsample_2
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

    N = 2.^randi(6);    
    m = randi(1000);
    
    x = m*ones(N * [1 1 1]);
    
    xx = Upsample_2(x);
    
    ok = mean(xx(:)) - m/8 < eps;
    
end