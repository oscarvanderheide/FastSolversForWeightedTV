function MRIrecon_TV(contrast, sampling, data, groundtruth, options, s_image, alphas, subfolder)
% MRIrecon_TV
%   MRIrecon_TV(contrast, sampling, data, groundtruth, options, s_image, alphas, subfolder)
% executes the MRI reconstruction with a total variation prior. The resuls 
% will be saved as files in the folders specified by "options" and if desired 
% in a specific "subfolder".
%
% Input:    
%   contrast [string]              
%       either 'T1' or 'T2'
%
%   sampling [string]
%       name of sampling. needs to be a name of a field of data.
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

    if nargin < 8; subfolder = '.'; end
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
    prior_options.name = 'TV';
    prior_options.s_image = s_image;

    % prepare MRI data
    MRI_data.s_kspace = s_image;
    f = eval(sprintf('data.mri.freq.%s', sampling));
    MRI_data.index = MRI_freq2ind(f, MRI_data.s_kspace);
    MRI_data.data = eval(sprintf('data.mri.%s.%s.data_noisy', contrast, sampling));

    param_str = 'alpha';
    param = alphas;

    u_all = cell(length(alphas),1);
    stats_all = cell(length(alphas),1);

    for a = 1 : length(alphas)
        prior_options.alpha = alphas(a);
        [u_all{a}, stats_all{a}] = MRI2_ADMM(MRI_data, ADMM_options, prior_options);
    end

    % save results
    folder = [options.misc.folder.results '/' subfolder '/' contrast '_' prior_options.name];
    filename = [folder '/' options.misc.name '_' contrast '_' sampling];
    mkdir(folder);
    gt = eval(sprintf('groundtruth.%s', contrast));
    [u_opt, i_opt] = MRIstats_one_param(u_all, gt, param, param_str, filename);
    stats = stats_all{i_opt};
    figure(101); clf; MRI2_ADMM_plot_stats(stats);
    saveas(gcf, [filename '_stats.png']);
    prior_options.prox_options.PC = [];
    save([filename '.mat'], 'u_opt', 'u_all', 'stats', 'ADMM_options', 'sampling', 'contrast', 'prior_options', 'MRI_data', 'alphas');

end