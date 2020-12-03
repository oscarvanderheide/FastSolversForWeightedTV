function [u, p_nn, stats] = FGP_wTV(data, alpha, n_iter, w, options, p_n)
% FGP_wTV
%   [u, p_nn, stats] = FGP_wTV(data, alpha, n_iter, options, p_nn) runs the fast 
% gradient projection algorithm on the weighted TV dual problem. The
% algorithm is adopted from the one for TV denoising published in [1].
%
% [1] Beck, A., & Teboulle, M. (2009). Fast gradient-based algorithms for 
%   constrained total variation image denoising and deblurring problems. 
%   IEEE Transactions on Image Processing, 18(11), 2419-2434. doi:10.1109/TIP.2009.2028250
%
% Input:    
%   data [matrix]              
%       noisy input image
%
%   alpha [float]       
%       regularization parameter
%
%   n_iter [int; DEFAULT = 20]
%       maximum number of iterations.
%
%   w [matrix]              
%       weights for weighted TV prior
%
%   options [struct; optional]            
%       there are many options of the algorithm which can be changed   
%
%       options.verbose [int; DEFAULT = 0]
%           verbosity level, the default is no output on the screen at all.
%       options.all [int; DEFAULT = 0]
%           save every options.all iterates in the cell arrais stats.all_u
%           and stats.all_Dp
%       options.tol_pn [scalar; DEFAULT = 0]
%           tolerance for convergence. The default does not check anything
%           as it slows done the computation
%    
%   p_n [matrix; DEFAULT = zeros]
%       initial guess for the dual state. I guess that any [NxMx2] matrix
%       whose norm over the third component is smaller than 1 can be taken
%       as an inital guess. It might be that any matrix can be used.
%    
% Output:
%   u [matrix]
%       denoised image
%
%   p_nn [matrix]
%       dual state of the final iterate
%
%   stats [struct; optional]
%       some statistics as function value for every iteration are stored in
%       this struct. The fields are
%
%       stats.fun
%           function value at any iteration
%       stats.du
%           largest change of any component
%       stats.du
%           largest change of the dual variable at any component
%       stats.time
%           largest change of any component at any iteration
%       stats.converged
%           number of iterations until convergence
%       stats.all_u
%           all iterates
%       stats.all_Dp
%           all dual variables of the iterates
%        
% See also:
%
% -------------------------------------------------------------------------
%   changes:
% 
% 2015-06-11 --------------------------------------------------------------
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
    
    s_data = size(data);

    if nargin < 4; n_iter = 20; end;
    if nargin < 6; p_n = zeros([s_data 2]); end;

    % treat the input options
    if ~exist('options', 'var'),            options = struct; end;
    if ~isfield(options, 'verbose'),        options.verbose         = 0;    end;
    if ~isfield(options, 'all'),            options.all             = 0;    end;
    if ~isfield(options, 'PC'),             options.PC              = @(x) x; end;

    if numel(options.PC) == 2
        PC = @(x) proj_box(x,options.PC{1},options.PC{2});
    else
        PC = options.PC;
    end
    
    debug = nargout > 2 || options.verbose > 0;
                
    % initialize
    factr = 1./(8*alpha);
    q = p_n; 
    p_nn = p_n;
    t_n = 1;

    if ~debug % for efficiency
        for n = 1:n_iter
            p_nn = PP(q + factr * G(PC(data + alpha*D(q,w)),q,w));
            
            if n == n_iter; break; end;
            
            t_nn = (1 + sqrt(1 + 4*t_n^2))./2;
            q = p_nn + (t_n-1)./t_nn.*(p_nn-p_n); 
            p_n = p_nn;
            t_n = t_nn;
        end        
        u = PC(data + alpha*D(p_nn,w));

    else % for debuggin
        stats.fun              = nan(n_iter,1);
        stats.dp               = nan(n_iter,1);
        stats.time             = nan(n_iter,1);
        stats.converged        = -1;
        if options.all > 0; 
            stats.all_u = cell(n_iter+1,1);
            stats.all_u{1} = zeros(s_data);
            stats.all_p = cell(n_iter+1,1);
            stats.all_p{1} = zeros(s_data);
        else
            stats.all_u = [];
            stats.all_p = [];
        end
        
        timer = tic;
        fun_0 = obj_fun(zeros(s_data),alpha,data, w);
        if nargout > 2; stats.fun(1) = fun_0; end;
        % display progress
        if options.verbose; fprintf('%s\n', Output_string(Function_name, 0, toc(timer), fun_0, nan)); end;
        
        for n = 1:n_iter
            p_nn = PP(q + factr * G(PC(data + alpha*D(q,w)),q,w));
            
            t_nn = (1 + sqrt(1 + 4*t_n^2))./2;
            q = p_nn + (t_n-1)./t_nn.*(p_nn-p_n);
            p_n = p_nn;
            t_n = t_nn;
            
            timer_n = toc(timer);
            u = PC(data + alpha*D(p_nn,w));
            fun_n = obj_fun(u, alpha, data, w);
            
            dp_max = norm(p_nn(:)-p_n(:),'inf');
            stats.dp(n) = dp_max;
            
            % display progress
            if options.verbose; fprintf('%s\n', Output_string(Function_name, n, timer_n, fun_n, dp_max)); end;
            
            if mod(n,options.all) == 0;
                stats.all_u{n+1} = u;
                stats.all_p{n+1} = p_nn;
            end
            stats.fun(n) = fun_n;
            stats.time(n) = timer_n;
        end
        u = PC(data + alpha*D(p_nn,w));
         
        if options.verbose; fprintf('%s: Iterations are done!\n', Function_name); end
    end
end

function x = G(x,y,w)
    x = Gradient2D_forward_constant_unitstep(x,y);
    x = Vec2D_mult(x,w);
end

function x = D(x,w)   
    x = Vec2D_mult(x,w);
    x = nAdj_Gradient2D_forward_constant_unitstep(x,w);
end

function x = PP(x)
    nx = max(1,sqrt(sum(x.^2,3)));
    x(:,:,1) = x(:,:,1) ./ nx;
    x(:,:,2) = x(:,:,2) ./ nx;
end

function y = proj_box(x,l,u)
    y = min(max(x,l),u);
end

function fun = obj_fun(x, alpha, data, w)
    % data
    res = x - data;
    f1 = 0.5 * sum(abs(res(:)).^2);
    
    % TV prior
    prior_options.alpha = alpha;
    prior_options.beta = 0;
    prior_options.s_voxel = 1;
    f2 = Penalty_wTV(prior_options, x, w);
    
    % sum both components up
    fun = f1 + f2;
end

function s = Output_string(fun_name, n, timer_n, fun_n, dp_max)    
    s = sprintf('   %s: step %4i: %5.1fs, fun=%8.6e, dp_max=%3.1e', fun_name, n, timer_n, fun_n, dp_max);   
end