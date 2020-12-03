function data_noisy = MRI_noise(data, sigma)
% MRI_noise
%   data_noisy = MRI_noise(data, sigma) adds standard normal distributed 
% complex valued noise.
%
% Input:
%   data [vector]
%       dimensions of the noise output
%
%   sigma [scalar, vector]
%       standard deviation for the noise distribution. If a scalar, then
%       the standard deviation is constant among the components
%
% Output:
%   data_noisy [vectro]
%       standard normal distributed complex noise
%
% See also:
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

    s_data = size(data);
    data_noisy = data + sigma .*  1./sqrt(2) .* (randn(s_data) + 1i .* randn(s_data));
    
end