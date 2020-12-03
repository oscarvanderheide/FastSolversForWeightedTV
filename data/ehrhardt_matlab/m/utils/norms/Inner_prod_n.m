function ip = Inner_prod_n(u, v, dim)
% Inner_prod_n
%   ip3 = Inner_prod_n(u, v, dim) computes the (Eucledian) inner product 
% (scalar product) of u and v at every pixel. The dimension dim is then one
% we take the sum over. 
%
% Input:
%   u [matrix]
%       input image 1
%       
%   v [matrix]
%       input image 2
%       
%   dim [int; DEFAULT=length(size(u))] 
%       dimension which shall be used for the inner product, see the text above.
%
% Output:
%   ip [matrix]
%       the output's dimension is the dimension of u and v reduced by 1.
%
% See also:
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

    if ~exist('dim', 'var'); dim = length(size(u)); end;
    
    ip = sum(u .*v, dim);
    
end