function data = MRI_sample_index_forward(index, kspace)
% MRI_sample_index_forward
%   data = MRI_sample_index_forward(index, kspace) samples the kspace at given 
% indecies.
%
% Input:
%   index [vector]
%       list of indecies
%
%   kspace [matrix]
%       full kspace data from each we wish to sample.
%
% Output:
%   data [vector]
%       vector containing the data at given points in kspace.
%
% See also: MRI_sample_index_backward
%
% -------------------------------------------------------------------------
%   changes:
% 
% 2015-06-04 --------------------------------------------------------------
% Matthias J. Ehrhardt
% CMIC, University College London, UK 
% matthias.ehrhardt.11@ucl.ac.uk
% http://www.cs.ucl.ac.uk/staff/ehrhardt/software.html
%
% -------------------------------------------------------------------------
% Copyright 2015 University College London
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
 
    data = Column(kspace(index));

end