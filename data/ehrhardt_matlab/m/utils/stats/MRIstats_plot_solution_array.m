function con_sol = MRIstats_plot_solution_array(solutions, err, min_i, param, param_str, backg)
% MRIstats_plot_solution_array
%   MRIstats_plot_solution_array(solutions, e, min_i, param, param_str, backg)
% plots an array of solutions with some statistics.
%
% Input:    
%   solutions [cell]              
%       array of solutions
%
%   err [vector]              
%       vector of quality measures for all solutions
%
%   min_i [int]              
%       index of best solution
%
%   param [vector]              
%       vector of parameters that have been tested
%
%   param_str [string]              
%       name of these parameters
%
%   backg [vector]              
%       background colour for text
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

    if nargin < 6; backg = [1 1 1]; end;
    
    n_rows = 3;
    n_cols = 3;
    i = find(~isnan(err));
    s = size(solutions{i(1)});
    r = sort(randperm(length(i),min(8,length(i))));
    
    con_sol = MRIstats_concat_solutions(solutions([min_i; i(r)]),n_rows,n_cols);
    
    imagesc(con_sol); axis image; colorbar;
    
    for n = 1 : n_rows
        for m = 1 : n_cols
            if n_cols*(n-1)+m-1 <= length(r)
                if n*m == 1
                    k = min_i;
                else
                    k = i(r(n_cols*(n-1)+m-1));
                end

                str = sprintf('%s:%2.1e,%3.1f', param_str, param(k), err(k));
                x = (m-1)*s(2)+10;
                y = (n-1)*s(1)+20;

                if numel(backg) > 0
                    text(x ,y, str, 'BackgroundColor', backg);
                else
                    text(x, y, str);
                end
            end
        end
    end
    
end