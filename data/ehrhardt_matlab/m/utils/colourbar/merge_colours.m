function c = merge_colours(tight, varargin)
% merge_colours
%   c = merge_colours(tight, varargin) merge any number of colourmaps.
%
% Input:
%   tight [boolean]
%       Shall the connection be tight? E.g. when connecting a map from red
%       to white and white to blue, shall white be used once (tight = true)
%       or twice (tight = false).
%
%   varargin [matrices]
%       Any number of colourmaps.
%
% Output:
%   c [Nx3 matrix]
%       merged colourmap
%        
% See also:
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

    K = nargin-1;
    
    N = - tight * (K-1);
    for k = 1 : K
        N = N + size(varargin{k},1);
    end
    
    c = zeros(N,3);
    
    v = varargin{1};
    i1 = 1;
    i2 = size(varargin{1},1);
    c(i1:i2,:) = v(1:end,:);
    
    for k = 2 : K
        v = varargin{k};
        i1 = i2 + 1;
        i2 = i2 + size(v,1)-tight;
        c(i1:i2,:) = v(1+tight:end,:);
    end
    
end