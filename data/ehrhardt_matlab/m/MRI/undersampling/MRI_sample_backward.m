function kspace = MRI_sample_backward(data, freq, s_kspace)
% MRI_sample_backward
%   kspace = MRI_sample_backward(data, freq, s_kspace) performs zero filling with 
% some sampled data.
%
% Input:
%   data [vector]
%       vector containing the data at given points in kspace.
%
%   freq [matrix]
%       matrix where each column are the frequencies of a data point in
%       kspace.
%
%   s_kspace [vector]
%       size of the output kspace.
%
% Output:
%   kspace [matrix]
%       full kspace data from each we wish to sample.
%
% See also: MRI_sample_forward
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

    ind = MRI_freq2ind(freq, s_kspace);
    kspace = reshape(accumarray(Column(ind), Column(data), [prod(s_kspace),1]), s_kspace);

end