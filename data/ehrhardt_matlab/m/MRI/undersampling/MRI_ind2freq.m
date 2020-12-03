function freq = MRI_ind2freq(ind, s_kspace)
% MRI_ind2freq
%   freq = MRI_ind2freq(ind, s_kspace) converts indices to frequencies
%
% Input:
%   ind [vector]
%       indices for data vector
%
%   s_kspace [vector]
%       size of the kspace
%
% Output:
%   freq [matrix]
%       locations in kspace
%
% See also: MRI_freq2ind
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

    ss_kspace = cumprod(s_kspace);
    dim = numel(s_kspace);
    
    freq = zeros(dim, length(ind)); 
        
    for i = numel(s_kspace) : -1 : 2
        freq(i,:) = fix((ind-1) / ss_kspace(i-1)) + 1;
        ind = mod(ind-1, ss_kspace(i-1))+1;
    end
      
    freq(1,:) = ind;
    
    for i = 1 : dim
        freq(i,:) = freq(i,:) - ceil((s_kspace(i)-1)/2) - 1;
    end
        
end