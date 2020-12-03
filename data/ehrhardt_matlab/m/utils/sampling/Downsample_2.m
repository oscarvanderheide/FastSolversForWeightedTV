function x = Downsample_2(X)
% Downsample_2
%   x = Downsample_2(X) samples an input image down by a factor of 2 via
% taking mean values.
%
% Input:    
%   X [matrix]              
%       scalar valued function
%
% Output:
%   x [matrix]
%       downscaled output
%
% See also: Downsample_N Upsample_2 Upsample_N
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

    if iscell(X)
        x = X;
        for i = 1 : length(x)
            x{i} = Downsample_2(x{i});
        end
    else   
%         if prod(size(X)-1) == 0
%             error('1 dimensional input is currently not supported');
% 
%         else
            switch length(size(X))
                case 2
                    x = X(1:2:end,1:2:end) + X(2:2:end,1:2:end) + X(1:2:end,2:2:end) + X(2:2:end,2:2:end);
%                     x = .25*x;
                case 3
                    x = X(1:2:end,1:2:end,1:2:end) + X(1:2:end,2:2:end,1:2:end) ...
                        + X(1:2:end,1:2:end,2:2:end) + X(1:2:end,2:2:end,2:2:end) ...
                        + X(2:2:end,1:2:end,1:2:end) + X(2:2:end,2:2:end,1:2:end) ...
                        + X(2:2:end,1:2:end,2:2:end) + X(2:2:end,2:2:end,2:2:end);
%                     x = .125*x;
                otherwise
                    error('%i dimensional input is currently not supported', length(size(X)));
            end
%         end

        x = x ./ (2^length(size(X)));

    end
    
end