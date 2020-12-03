function [conv, time, sol, param, parts, all] = Parameter_estimation_realdata(...
        fun, parallel, varargin)
% Parameter_estimation_realdata
%   [conv, time, sol, param, parts, all] = Parameter_estimation_realdata(...
%       fun, parallel, varargin)
%
% Input:
%   fun [function handle]
%       solution operator
%
%   parallel [int] 
%       number of workers
%
%   varargin [vectors] 
%       all parameters
%
% Output:
%   conv [matrix]
%       converged?
%
%   time [matrix]
%       time for each run
%
%   sol [cell]
%       solution for each run
%
%   param [cell]
%       parameters for each run
%
%   parts [cell]
%       parts for each run
%
%   all [cell]
%       iterates for each run
%
% See also: Parameter_estimation
%
% -------------------------------------------------------------------------
%   changes:
% 
% 2015-01-05 --------------------------------------------------------------
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
        
    [param, err] = Prepare_parameters(varargin);
        
    conv = nan(size(err));
    time = nan(size(err));
    parts = cell(size(err));
    
    store_all = nargout > 5;
    if store_all
        all = cell(size(param, 1), 1);
    end
    
    store_solution = nargout > 2;
    if store_solution; sol = cell(size(param, 1), 1); end;
        
    if parallel > 1
        if matlabpool('size') == 0; 
            matlabpool(parallel)
        end
        
        parfor p = 1 : size(param, 1)
            param_p = param(p,:);
            
            pp = tic;
            if store_all
                [sol_p, conv_p, parts_p, all_p] = fun(param_p);
                all{p} = all_p;
            else
                [sol_p, conv_p, parts_p] = fun(param_p);
            end
            time(p) = toc(pp);
            
            conv(p) = conv_p;
            parts{p} = parts_p;
            
            if store_solution; sol{p} = sol_p; end;
        end
        
        matlabpool close;
    else
        for p = 1 : size(param, 1)
            param_p = param(p,:);
            
            pp = tic;
            if store_all
                [sol_p, conv_p, parts_p, all_p] = fun(param_p);
                all{p} = all_p;
            else
                [sol_p, conv_p, parts_p] = fun(param_p);
            end
            time(p) = toc(pp);
            
            conv(p) = conv_p;
            parts{p} = parts_p;
            
            if store_solution; sol{p} = sol_p; end;
        end
    end
    
    parts = ConvertErrorMat(parts); 
    
end