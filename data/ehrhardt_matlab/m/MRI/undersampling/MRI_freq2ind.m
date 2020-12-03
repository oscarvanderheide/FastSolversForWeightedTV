function ind = MRI_freq2ind(freq, s_kspace)
% MRI_freq2ind
%   ind = MRI_freq2ind(freq, s_kspace) converts frequencies to indices in
%   kspace.
%
% Input:
%   freq [matrix]
%       locations in kspace
%
%   s_kspace [vector]
%       size of the kspace
%
% Output:
%   ind [vector]
%       indices corresponding to the input frequencies
%
% See also: MRI_ind2freq
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

    dim = size(freq, 1);
    ss_kspace = cumprod(s_kspace);
    
    for i = 1 : dim
        freq(i,:) = freq(i,:) + ceil((s_kspace(i)-1)/2) + 1;
    end
    
    ind = freq(1,:);
    
    for i = 2 : dim
        ind = ind + ss_kspace(i-1)*(freq(i,:)-1);
    end
    
end