function con_sol = MRIstats_concat_solutions(solutions, N, M)
% MRIstats_concat_solutions
%   con_sol = MRIstats_concat_solutions(solutions, N, M) concatinates the
% solutions from the cell "solutions" into a big image with N rows and M
% columns.
%
% Input:    
%   solutions [cell]              
%       array of solutions
%
%   N [int]              
%       number of rows
%
%   M [int]              
%       number of columns
%
% Output:    
%   con_sol [matrix]              
%       image with all solutions
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
    
    if nargin < 2; N = size(solutions,1); M = size(solutions,2); end;
    
    s = size(solutions{1});
    con_sol = nan(M*s(1),N*s(2));
    
    for n = 1 : N
        for m = 1 : M
            if (N*(m-1) + n) <= numel(solutions)
                x = ((m-1)*s(1)+1):m*s(1);
                y = ((n-1)*s(2)+1):n*s(2);
                
                con_sol(x,y) = solutions{N*(m-1) + n};
            end
        end
    end
    
end