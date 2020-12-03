function [u, stats] = MRI2_ADMM(MRI_data, ADMM_options, prior_options)
% MRI2_ADMM
%   [u, stats] = MRI2_ADMM(MRI_data, ADMM_options, prior_options) runs
% an ADMM algorithm to reconstruct MRI with a non-differentiable prior.
%
% Input:    
%   MRI_data [struct]              
%       struct of MRI data. The fields are
%           MRI_data.data [complex vector]        
%               actual imaging data
%           MRI_data.index [int vector] 
%               the indices where the data was sampled
%           MRI_data.s_kspace [int vector]
%               the size of the kspace
%
%   ADMM_options [struct]
%       struct of ADMM options. The fields are
%           ADMM_options.init [matrix; DEFAULT = zeros]
%               inital guess of the image
%           ADMM_options.n_iter [int; DEFAULT = 10] 
%               number of iterations 
%           ADMM_options.rho [double; DEFAULT = 1]       
%               step size in ADMM
%           ADMM_options.rho_dynamic [boolean; DEFAULT = true]
%               shall rho be chosen dynamically?
%           ADMM_options.mu [double; DEFAULT = 5]
%               if the primal residual is mu times larger or small than the 
%               dual residual, adapt rho by tau_incr / tau_decr 
%           ADMM_options.tau_incr [double; DEFAULT = 2]
%               increasing factor for rho if the residuals are too far apart
%           ADMM_options.tau_decr [double; DEFAULT = 3]
%               decreasing factor for rho if the residuals are too far apart
%           ADMM_options.all [int; DEFAULT = 0]
%               save every all-nth iteration in stats.all
%           ADMM_options.GPU [boolean; DEFAULT = false]
%               use GPU?
%
%   prior_options [struct]              
%       struct of prior options. The fields are
%
%           prior_options.name [string]        
%               name of the prior. Currently supported: 'TV', 'wTV', 'dTV',
%               'Tik'
%
%           prior_options.alpha [double >= 0] 
%               regularization parameter
%
%           prior_options.prox_options [struct; DEFAULT = struct] 
%               struct with options for the proximity operator. Fields are
%
%                   prox_options.n_iter [int; DEFAULT = 1]
%                       number of steps in the iterative estimation of the
%                       proximity operator.
%    
% Output:
%   u [vector]
%       final iterate
%
%   stats [struct; optional]
%       some statistics as function value for every iteration are stored in
%       this struct. The fields are
%
%       stats.fun
%           function value at any iteration
%       stats.data_fit
%           data fit at any iteration
%       stats.prior
%           prior value at any iteration
%       stats.constraint1
%           value of the first constraint at any iteration
%       stats.constraint2
%           dvalue of the second constraint at any iteration
%       stats.res_primal
%           primal residual at every iteration
%       stats.res_dual
%           dual residual at every iteration
%       stats.rho
%           rho value at any iteration
%       stats.time
%           time at every iteration
%       stats.all
%           cell array with iterates
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
        
    % treat the input options
    if ~exist('ADMM_options', 'var'), ADMM_options = struct; end; 
    if ~isfield(ADMM_options, 'init'),              ADMM_options.init               = []; end;
    if ~isfield(ADMM_options, 'n_iter'),            ADMM_options.n_iter             = 10; end;
    if ~isfield(ADMM_options, 'rho'),               ADMM_options.rho                = 1; end;
    if ~isfield(ADMM_options, 'rho_dynamic'),       ADMM_options.rho_dynamic        = true; end;
    if ~isfield(ADMM_options, 'mu'),                ADMM_options.mu                 = 5; end;
    if ~isfield(ADMM_options, 'tau_incr'),          ADMM_options.tau_incr           = 2; end;
    if ~isfield(ADMM_options, 'tau_decr'),          ADMM_options.tau_decr           = 3; end;
    if ~isfield(ADMM_options, 'all'),               ADMM_options.all                = 0; end;
    if ~isfield(ADMM_options, 'GPU'),               ADMM_options.GPU                = 0; end;
    
    if ~isfield(prior_options, 'prox_options'),         prior_options.prox_options        = struct; end;
    if ~isfield(prior_options.prox_options, 'n_iter'),  prior_options.prox_options.n_iter = 1; end;
    
    s_image = prior_options.s_image;
    rho = ADMM_options.rho;
    K = ADMM_options.n_iter;

    debug = nargout > 1;
    
    if debug
        stats.fun = nan(K,1);
        stats.data_fit = nan(K,1);
        stats.prior = nan(K,1);
        stats.constraint1 = nan(K,1);
        stats.constraint2 = nan(K,1);
        stats.res_primal = nan(K,1); 
        stats.res_dual = nan(K,1);
        stats.rho = nan(K,1);
        stats.time = nan(K,1);
        stats.all = cell(K,1);
    end
        
    if ADMM_options.GPU; 
        grad = zeros([prior_options.s_image 2], 'gpuArray');
        p_k = zeros(size(grad), 'gpuArray');
    else
        grad = zeros([prior_options.s_image 2]);
        p_k = zeros(size(grad));
    end
               
    switch prior_options.name
        case {'TV', 'Tik'}
            prior_options.sideinfo = 0;
    end
    
    if ADMM_options.GPU; prior_options.sideinfo = gpuArray(prior_options.sideinfo); end
    
        switch prior_options.name
            case 'TV'
                prox = @(x,alpha,p) FGP_TV(x, alpha, prior_options.prox_options.n_iter, prior_options.prox_options, p);
            case 'wTV'
                prox = @(x,alpha,p) FGP_wTV(x, alpha, prior_options.prox_options.n_iter, prior_options.sideinfo, prior_options.prox_options,p);
            case 'dTV'
                prox = @(x,alpha,p) FGP_dTV(x, alpha, prior_options.prox_options.n_iter, prior_options.sideinfo, prior_options.prox_options, p);
            case 'Tik'
                prox = @(x,alpha) x./(1+alpha);
            otherwise
                error('prior %s not yet implemented', prior_options.name);
        end
        
    % set operators
    S = @(x) MRI_sample_index_forward(MRI_data.index, x); % sampling
    Sadj = @(x) MRI_sample_index_backward(x, MRI_data.index, MRI_data.s_kspace); % backwards sampling
    F = @(x) Ufftn(x); % Fourier operator
    Fadj = @(x) real(Uifftn(x)); % inverse Fourier operator

    % initialize
    ones_array = ones(MRI_data.s_kspace);
    if ADMM_options.GPU
        zeros_array = zeros(s_image, 'gpuArray');
    else
        zeros_array = zeros(s_image);
    end
    
    D = (Sadj(S(ones_array)) + rho).^(-1); % (number of sample at this location + rho) inverse
    if ADMM_options.GPU; D = gpuArray(D); end;
    b = Sadj(MRI_data.data);
    if ADMM_options.GPU; b = gpuArray(b); end;
    
    if numel(ADMM_options.init) == 0;
        z = zeros_array; %Fadj(f);
        f = zeros_array; %D.*b;
    else
        z = ADMM_options.init;
        f = F(z);
    end
    
    y1 = zeros_array;
    y2 = zeros_array;
    if ADMM_options.GPU; MRI_data.data = gpuArray(MRI_data.data); end;
    
    if ~debug % for speed
        for k = 1:K
            % First block
            switch prior_options.name
                case {'Tik'}
                    u = prox(z - y2/rho, prior_options.alpha/rho);
                case {'TV', 'wTV','dTV'}
                    [u, p_k] = prox(z - y2/rho, prior_options.alpha/rho, p_k);
                otherwise
                    error('Prior not yet supported: %s', prior_options.name);
            end
            
            x = D .* (b + rho*f - y1);
            
            % Second block
            z_ = 1./2 * (Fadj(x + y1/rho) + u + y2/rho);
            f = F(z_);

            % update multipliers
            d1 = x - f;
            d2 = u - z_;
            y1 = y1 + rho*d1;
            y2 = y2 + rho*d2;
            
            % calculate residuals
            s = sqrt(2)*rho*norm(z_(:)-z(:));
            r = sqrt(norm(d1(:))^2 + norm(d2(:))^2);

            % update
            if ADMM_options.rho_dynamic
                [rho, updated] = Update_rho(rho, r, s, ...
                    ADMM_options.mu, ADMM_options.tau_incr, ADMM_options.tau_decr);
                if updated; 
                    D = (Sadj(S(ones_array)) + rho).^(-1); 
                    if ADMM_options.GPU; D = gpuArray(D); end;
                end;
            end
            z = z_;
        end
    else % for debugging
        tic;
        
        for k = 1 : K
            
            % First block
            switch prior_options.name
                case {'Tik'}
                    u = prox(z - y2/rho, prior_options.alpha/rho);
                case {'TV', 'wTV','dTV'}
                    [u, p_k] = prox(z - y2/rho, prior_options.alpha/rho, p_k);
                otherwise
                    error('Prior not yet supported: %s', prior_options.name);
            end
            
            x = D .* (b + rho*f - y1);
     
            % Second block
            z_ = 1./2 * (Fadj(x + y1/rho) + u + y2/rho);
            f = F(z_);

            % update multipliers
            d1 = x - f;
            d2 = u - z_;
            y1 = y1 + rho*d1;
            y2 = y2 + rho*d2;
     
            if mod(k,ADMM_options.all) == 0
                if ADMM_options.GPU
                    stats.all{k} = gather(u);
                else
                    stats.all{k} = u;
                end
            end
            
            % calculate residuals
            stats.res_dual(k) = sqrt(2)*rho*norm(z_(:)-z(:));
            stats.res_primal(k) = sqrt(norm(d1(:))^2 + norm(d2(:))^2);
            
            % calculate objective
            if ADMM_options.GPU
                [stats.fun(k), stats.data_fit(k), stats.prior(k), stats.constraint1(k), stats.constraint2(k)] = ...
                    obj(S(x),MRI_data.data,u,...
                    prior_options.alpha,prior_options.name,rho,...
                    x,f,y1,z_,y2,grad,prior_options.sideinfo);
            else
                [stats.fun(k), stats.data_fit(k), stats.prior(k), stats.constraint1(k), stats.constraint2(k)] = ...
                    obj(S(x),MRI_data.data,u,prior_options.alpha,prior_options.name,rho,x,f,y1,z_,y2,grad,prior_options.sideinfo);
            end
        
            % update
            stats.rho(k) = rho;
            if ADMM_options.rho_dynamic
                [rho, updated] = Update_rho(rho, stats.res_primal(k), stats.res_dual(k), ...
                    ADMM_options.mu, ADMM_options.tau_incr, ADMM_options.tau_decr);
                if updated; D = (Sadj(S(ones_array)) + rho).^(-1); end;
            end
            z = z_;
            
            stats.time(k) = toc;
        end
    end
    if ADMM_options.GPU; u = gather(u); end
end
            
function [fun, data_fit, prior, constraint1, constraint2] = obj(Sx,b,u,alpha,name,rho,x,Fz,y1,z,y2,grad,side_info)
    
    options.alpha = alpha;
    options.s_image = size(u);
    options.s_voxel = [1 1];
    
    switch name
        case 'TV'
            prior = Penalty_TV(options,u,grad);
        
        case 'wTV'
            prior = Penalty_wTV(options,u,side_info,grad);
        
        case 'dTV'
            prior = Penalty_dTV(options,u,side_info,grad);
        
        case 'Tik'
            prior = alpha*sum(u(:).^2);

        otherwise
            error('prior %s not yet defined', name);
    end
    
    data_fit = .5* sum(abs(Sx(:)-b(:)).^2);
    constraint1 = real(y1(:)'*(x(:)+Fz(:))) + rho/2 * sum(abs(x(:)-Fz(:)).^2);
    constraint2 = y2(:)'*(u(:)+z(:)) + rho/2 * sum((u(:)-z(:)).^2);
            
    data_fit = gather(data_fit);    
    prior = gather(prior); 
    constraint1 = gather(constraint1);    
    constraint2 = gather(constraint2);    

    fun = data_fit + prior + constraint1 + constraint2;    
end

function [rho, updated] = Update_rho(rho, r, s, mu, tau_incr, tau_decr)
    updated = false;

    if r > mu*s
        rho = rho * tau_incr;
        updated = true;
    else
        if s > mu*r
            rho = rho / tau_decr;
            updated = true;
        end
    end
end