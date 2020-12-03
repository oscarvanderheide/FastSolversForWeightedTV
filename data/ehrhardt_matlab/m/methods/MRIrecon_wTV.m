function MRIrecon_wTV(contrast, sampling, sideinfo_filename, sideinfo_str, sideinfo, data, groundtruth, options, s_image, alphas, etas, subfolder)
% MRIrecon_wTV
%   MRIrecon_wTV(contrast, sampling, sideinfo_filename, sideinfo_str, sideinfo, data, groundtruth, options, s_image, alphas, etas, subfolder)
% executes the MRI reconstruction with a weighted total variation prior. 
% The resuls will be saved as files in the folders specified by "options" 
% and if desired in a specific "subfolder".
%
% Input:    
%   contrast [string]              
%       either 'T1' or 'T2'
%
%   sampling [string]
%       name of sampling. needs to be a name of a field of data.
%
%   sideinfo_filename [string]
%       one can pass info about the side information
%
%   sideinfo_str [string]
%       one can pass info about the side information
%
%   sideinfo [matrix]
%       image from side information
%
%   data [struct]
%       data struct
%
%   groundtruth [struct]
%       ground truth struct
%
%   options [struct]
%       struct of options that has been generated with the appropriate
%       m-files.
%
%   s_image [int; DEFAULT = size of ground truth]
%       size of the image
%
%   alphas [vector]
%       vector of regularization parameters
%
%   etas [vector]
%       vector of edge parameters
%    
%   subfolder [matrix; DEFAULT = zeros]
%       initial guess for the dual state. Any [NxMx2] matrix whose norm over 
%       the third component is smaller than 1 can be taken as an inital guess.
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

    if nargin < 12; subfolder = '.'; end
    if isempty(s_image); s_image = options.mri.objectsize_in_pixel; end;

    % prepare ADMM options
    ADMM_options.n_iter = 300;
    ADMM_options.rho = 1;
    ADMM_options.rho_dynamic = 1;
    ADMM_options.mu = 5;
    ADMM_options.tau_incr = 2;
    ADMM_options.tau_decr = 3;
    ADMM_options.GPU = 0;
    ADMM_options.all = 0;
    ADMM_options.verbose = false;

    folder = [options.misc.folder.results '/' subfolder '/' contrast '_early_stopping'];
    filename = [folder '/' options.misc.name '_' contrast '_' sampling];
    tmp = load(filename, 'u_opt');
    ADMM_options.init = tmp.u_opt;

    prior_options.prox_options.PC = @(x) max(x,0);
    prior_options.prox_options.n_iter = 5;

    % prepare prior options
    prior_options.name = 'wTV';
    prior_options.s_image = s_image;
    prior_options.sideinfo_string = sideinfo_str;
    prior_options.sideinfo_filename = sideinfo_filename;
    g = Gradient2D_forward_constant_unitstep(sideinfo);
    n = Norm_n(g,0,3);

    ne = @(n, eta) sqrt(n.^2 + eta^2);
    si = @(n, eta) eta./ne(n, eta);

    % prepare MRI data
    MRI_data.s_kspace = s_image;
    f = eval(sprintf('data.mri.freq.%s', sampling));
    MRI_data.index = MRI_freq2ind(f, MRI_data.s_kspace);
    MRI_data.data = eval(sprintf('data.mri.%s.%s.data_noisy', contrast, sampling));

    u_all = cell(length(alphas),length(etas));
    stats_all = cell(length(alphas),length(etas));

    for e = 1 : length(etas)
        prior_options.sideinfo = si(n,etas(e));
        for a = 1 : length(alphas)
            prior_options.alpha = alphas(a);
            [u_all{a,e}, stats_all{a,e}] = MRI2_ADMM(MRI_data, ADMM_options, prior_options);     
        end
    end

    % save results
    folder = [options.misc.folder.results '/' subfolder '/' contrast '_' prior_options.name '_' prior_options.sideinfo_string];
    filename = [folder '/' options.misc.name '_' contrast '_' sampling];
    mkdir(folder);
    gt = eval(sprintf('groundtruth.%s', contrast));

    if numel(alphas) > 1 && numel(etas) > 1
        param{1} = alphas;
        param{2} = etas;
        param_str{1} = 'alpha';
        param_str{2} = 'eta';
        [u_opt, i_opt, j_opt] = MRIstats_two_param(u_all, gt, param, param_str, filename);
        stats = stats_all{i_opt,j_opt};
        eta_opt = etas(j_opt);
    else
        if numel(etas) > 1
            param = etas;
            param_str = 'eta';
            [u_opt, i_opt] = MRIstats_one_param(u_all, gt, param, param_str, filename);    
            stats = stats_all{i_opt};
            eta_opt = etas(i_opt);
        else
            param = alphas;
            param_str = 'alpha';
            [u_opt, i_opt] = MRIstats_one_param(u_all, gt, param, param_str, filename);    
            stats = stats_all{i_opt};
            eta_opt = etas;
        end
    end
    figure(101); clf; MRI2_ADMM_plot_stats(stats);
    saveas(gcf, [filename '_stats.png']);
    prior_options.prox_options.PC = [];
    save([filename '.mat'], 'u_opt', 'u_all', 'stats', 'ADMM_options', 'sampling', 'contrast', 'prior_options', 'MRI_data', 'alphas', 'etas');

    % save best side info
    prior_options.sideinfo = si(n,eta_opt);
    C = 256;
    imwrite((C-1)*prior_options.sideinfo, colormap(gray(C)), [filename '_sideinfo.png']);

end