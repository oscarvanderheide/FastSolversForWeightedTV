function X = Upsample_2(x)
% Upsample_2
%   X = Upsample_2(x) samples an input image up by a factor of 2 with
% constant values in the areas. Its scaled to be the adjoint of
% Downsample_2.
%
% Input:    
%   x [matrix]              
%       scalar valued function
%
% Output:
%   X [matrix]
%       upscaled output
%
% See also: Upsample_N Downsample_2 Downsample_N
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

    if iscell(x)
        X = x;
        for i = 1 : length(X)
            X{i} = Upsample_2(X{i});
        end
    else
%         if prod(size(X)-1) == 0
% 
% 
%         else
            x = x ./ (2^length(size(x)));            
            X = zeros(size(x)*2);
            
            switch length(size(X))
                case 2
%                     x = .25*x;
                    X(1:2:end,1:2:end) = x;
                    X(2:2:end,1:2:end) = x;
                    X(1:2:end,2:2:end) = x;
                    X(2:2:end,2:2:end) = x;
                case 3
%                     x = .125*x;
                    X(1:2:end,1:2:end,1:2:end) = x;
                    X(1:2:end,2:2:end,1:2:end) = x;
                    X(1:2:end,1:2:end,2:2:end) = x;
                    X(1:2:end,2:2:end,2:2:end) = x;
                    X(2:2:end,1:2:end,1:2:end) = x;
                    X(2:2:end,2:2:end,1:2:end) = x;
                    X(2:2:end,1:2:end,2:2:end) = x;
                    X(2:2:end,2:2:end,2:2:end) = x;
                otherwise
                    error('%i dimensional input is currently not supported', length(size(X)));
            end
%         end      
        
    end
end