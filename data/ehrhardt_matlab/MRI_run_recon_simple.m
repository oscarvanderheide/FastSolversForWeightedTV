% MRI_run_recon_BrainWebB
%   This script runs several reconstruction procedures for "BrainWebB"
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

%% load data
name = 'BrainWebB';
load(['data/data_paper/' name '/' name '.mat']);
s_image = options.mri.objectsize_in_pixel;

%% set parameters
alpha = 5e-3;
eta = 1e-2;
eta_large = 1e-1; % for visualization

%% set ADMM options
ADMM_options.n_iter = 300;
ADMM_options.rho = 1;
ADMM_options.rho_dynamic = 1;
ADMM_options.mu = 5;
ADMM_options.tau_incr = 2;
ADMM_options.tau_decr = 3;
ADMM_options.GPU = 0;
ADMM_options.all = 0;
ADMM_options.init = zeros(s_image);

%% prepare MRI data
MRI_data.s_kspace = s_image;
MRI_data.index = MRI_freq2ind(data.mri.freq.cartesianX7, MRI_data.s_kspace);
MRI_data.data = data.mri.T2.cartesianX7.data_noisy;

%% set general prior values
g = Gradient2D_forward_constant_unitstep(groundtruth.T1);
n = Norm_n(g,0,3);
prior_options.s_image = s_image;

%% early stopping
% prior options
prior_options.name = 'TV';
prior_options.sideinfo = zeros(s_image);
prior_options.alpha = 0;

% prox options
prior_options.prox_options.PC = @(x) max(x,0);
prior_options.prox_options.n_iter = 0;

%% run recon
tic; u_es = MRI2_ADMM(MRI_data, ADMM_options, prior_options); toc;

%% TV
% prior options
prior_options.name = 'TV';
prior_options.sideinfo = zeros(s_image);
prior_options.alpha = alpha;

% prox options
prior_options.prox_options.PC = @(x) max(x,0);
prior_options.prox_options.n_iter = 5;

%% run recon
tic; u_tv = MRI2_ADMM(MRI_data, ADMM_options, prior_options); toc;

%% wTV
% prior options
prior_options.name = 'wTV';
prior_options.sideinfo = eta./sqrt(n.^2 + eta^2);
prior_options.alpha = alpha;

% prox options
prior_options.prox_options.PC = @(x) max(x,0);
prior_options.prox_options.n_iter = 5;

%% run recon
tic; u_wTV = MRI2_ADMM(MRI_data, ADMM_options, prior_options); toc;

%% dTV
% prior options
prior_options.name = 'dTV';
prior_options.sideinfo = Vec2D_mult(g,1./sqrt(n.^2 + eta.^2));
prior_options.alpha = alpha;

% prox options
prior_options.prox_options.PC = @(x) max(x,0);
prior_options.prox_options.n_iter = 5;

%% run recon
tic; u_dTV = MRI2_ADMM(MRI_data, ADMM_options, prior_options); toc;

%% show problem
figure(1); clf;
subplot(2,2,1); imagesc(groundtruth.T2, [0,1]); axis image; colormap gray; title('ground truth');
sampling = zeros(s_image);
sampling(MRI_data.index) = 1;
subplot(2,2,2); imagesc(sampling); axis image; colormap gray; title('sampling');
subplot(2,2,3); imagesc(groundtruth.T1, [0,1]); axis image; colormap gray; title('other contrast');

%% show results
figure(2); clf;
subplot(2,2,1); imagesc(u_es, [0,1]); axis image; colormap gray; title('result early stopping');
subplot(2,2,2); imagesc(u_tv, [0,1]); axis image; colormap gray; title('result TV');
subplot(2,2,3); imagesc(u_wTV, [0,1]); axis image; colormap gray; title('result wTV');
subplot(2,2,4); imagesc(u_dTV, [0,1]); axis image; colormap gray; title('result dTV');

%% show results cropped
i_crop = 50:100; j_crop = 50:100;
figure(3); clf;
subplot(2,2,1); imagesc(u_es(i_crop,j_crop), [0,1]); axis image; colormap gray; title('result early stopping');
subplot(2,2,2); imagesc(u_tv(i_crop,j_crop), [0,1]); axis image; colormap gray; title('result TV');
subplot(2,2,3); imagesc(u_wTV(i_crop,j_crop), [0,1]); axis image; colormap gray; title('result wTV');
subplot(2,2,4); imagesc(u_dTV(i_crop,j_crop), [0,1]); axis image; colormap gray; title('result dTV');

%% show side info
figure(4); clf
p = eta_large./sqrt(n.^2 + eta_large^2);
subplot(1,2,1); imagesc(1-p); axis image; colormap gray; title('side info wTV');
p = Vec2D_mult(g,1./sqrt(n.^2 + eta_large.^2));
subplot(1,2,2); Image_complex(p(:,:,1)+1i*p(:,:,2),[],[],[],pi); axis image; title('side info'); axis image; title('side info dTV');
colormap gray;

%% save figures
for i = 1 : 4
    figure(i); saveas(gcf, sprintf('MRI_run_recon_simple_%i.png', i));
end