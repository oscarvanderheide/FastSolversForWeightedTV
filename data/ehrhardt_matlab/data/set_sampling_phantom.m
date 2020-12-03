% set_sampling_phantom.m
%   creates the sampling pattern for the phantom data
%
% See also: 
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

%% sampling
clear data
center_block = 16 * [1 1];

% cartesian standard x
options.mri.sampling.cartesianX_full.coverage = options.mri.objectsize_in_pixel;
options.mri.sampling.cartesianX_lowres.coverage = center_block;
options.mri.sampling.cartesianX_1d6.coverage = round(options.mri.objectsize_in_pixel .* [1 1./6]);

options.mri.sampling.cartesianX9.coverage = options.mri.objectsize_in_pixel;
options.mri.sampling.cartesianX9.every_nth_line = 9;
options.mri.sampling.cartesianX9.center_block = center_block;

options.mri.sampling.cartesianX7.coverage = options.mri.objectsize_in_pixel;
options.mri.sampling.cartesianX7.every_nth_line = 7;
options.mri.sampling.cartesianX7.center_block = center_block;

data.mri.freq.cartesianX_full = MRI2_sample_cartesian_x_standard(options.mri.sampling.cartesianX_full.coverage);
data.mri.freq.cartesianX_lowres = MRI2_sample_cartesian_x_standard(options.mri.sampling.cartesianX_lowres.coverage);
data.mri.freq.cartesianX_1d6 = MRI2_sample_cartesian_x_standard(options.mri.sampling.cartesianX_1d6.coverage);
data.mri.freq.cartesianX9 = MRI2_sample_cartesian_x_standard(options.mri.sampling.cartesianX9.coverage, [], options.mri.sampling.cartesianX9.every_nth_line);
data.mri.freq.cartesianX9 = [data.mri.freq.cartesianX9 MRI2_sample_cartesian_x_standard(options.mri.sampling.cartesianX9.center_block)];
data.mri.freq.cartesianX7 = MRI2_sample_cartesian_x_standard(options.mri.sampling.cartesianX7.coverage, [], options.mri.sampling.cartesianX7.every_nth_line);
data.mri.freq.cartesianX7 = [data.mri.freq.cartesianX7 MRI2_sample_cartesian_x_standard(options.mri.sampling.cartesianX7.center_block)];

options.mri.sampling.cartesianY11.coverage = options.mri.objectsize_in_pixel;
options.mri.sampling.cartesianY11.every_nth_line = 11;
options.mri.sampling.cartesianY11.center_block = center_block;

options.mri.sampling.cartesianY8.coverage = options.mri.objectsize_in_pixel;
options.mri.sampling.cartesianY8.every_nth_line = 8;
options.mri.sampling.cartesianY8.center_block = center_block;

data.mri.freq.cartesianY11 = MRI2_sample_cartesian_y_standard(options.mri.sampling.cartesianY11.coverage, [], options.mri.sampling.cartesianY11.every_nth_line);
data.mri.freq.cartesianY11 = [data.mri.freq.cartesianY11 MRI2_sample_cartesian_y_standard(options.mri.sampling.cartesianY11.center_block)];
data.mri.freq.cartesianY8 = MRI2_sample_cartesian_y_standard(options.mri.sampling.cartesianY8.coverage, [], options.mri.sampling.cartesianY8.every_nth_line);
data.mri.freq.cartesianY8 = [data.mri.freq.cartesianY8 MRI2_sample_cartesian_y_standard(options.mri.sampling.cartesianY8.center_block)];

% cartesian random x
cart_random = [0 4; 0 8; 0 8; 0 16; 0 32; 0 64];
for i = 1 : size(cart_random,1)
    eval(sprintf('options.mri.sampling.cartesianX_random_%i_%i.coverage = options.mri.objectsize_in_pixel;', cart_random(i,1),cart_random(i,2)))
    eval(sprintf('options.mri.sampling.cartesianX_random_%i_%i.n_fixed_lines = %i;', cart_random(i,1),cart_random(i,2),cart_random(i,1)))
    eval(sprintf('options.mri.sampling.cartesianX_random_%i_%i.n_random_lines = %i;', cart_random(i,1),cart_random(i,2),cart_random(i,2)))
    eval(sprintf('options.mri.sampling.cartesianX_random_%i_%i.center_block = center_block;', cart_random(i,1),cart_random(i,2)))
    eval(sprintf('data.mri.freq.cartesianX_random_%i_%i = MRI2_sample_cartesian_x_random_quadratic(options.mri.sampling.cartesianX_random_%i_%i.coverage, [], options.mri.sampling.cartesianX_random_%i_%i.n_fixed_lines, options.mri.sampling.cartesianX_random_%i_%i.n_random_lines);', cart_random(i,1),cart_random(i,2),cart_random(i,1),cart_random(i,2),cart_random(i,1),cart_random(i,2),cart_random(i,1),cart_random(i,2)))
    eval(sprintf('data.mri.freq.cartesianX_random_%i_%i = [data.mri.freq.cartesianX_random_%i_%i MRI2_sample_cartesian_x_standard(options.mri.sampling.cartesianX_random_%i_%i.center_block)];',cart_random(i,1),cart_random(i,2),cart_random(i,1),cart_random(i,2),cart_random(i,1),cart_random(i,2)))
end

% cartesian random y
cart_random = [0 4; 0 8; 0 16; 0 32; 0 64];
for i = 1 : size(cart_random,1)
    eval(sprintf('options.mri.sampling.cartesianY_random_%i_%i.coverage = options.mri.objectsize_in_pixel;', cart_random(i,1),cart_random(i,2)))
    eval(sprintf('options.mri.sampling.cartesianY_random_%i_%i.n_fixed_lines = %i;', cart_random(i,1),cart_random(i,2),cart_random(i,1)))
    eval(sprintf('options.mri.sampling.cartesianY_random_%i_%i.n_random_lines = %i;', cart_random(i,1),cart_random(i,2),cart_random(i,2)))
    eval(sprintf('options.mri.sampling.cartesianY_random_%i_%i.center_block = center_block;', cart_random(i,1),cart_random(i,2)))
    eval(sprintf('data.mri.freq.cartesianY_random_%i_%i = MRI2_sample_cartesian_y_random_quadratic(options.mri.sampling.cartesianY_random_%i_%i.coverage, [], options.mri.sampling.cartesianY_random_%i_%i.n_fixed_lines, options.mri.sampling.cartesianY_random_%i_%i.n_random_lines);', cart_random(i,1),cart_random(i,2),cart_random(i,1),cart_random(i,2),cart_random(i,1),cart_random(i,2),cart_random(i,1),cart_random(i,2)))
    eval(sprintf('data.mri.freq.cartesianY_random_%i_%i = [data.mri.freq.cartesianY_random_%i_%i MRI2_sample_cartesian_y_standard(options.mri.sampling.cartesianY_random_%i_%i.center_block)];',cart_random(i,1),cart_random(i,2),cart_random(i,1),cart_random(i,2),cart_random(i,1),cart_random(i,2)))
end

% radial standard
rad = [64; 32; 16; 8; 4];
for i = 1 : size(rad,1)
    eval(sprintf('options.mri.sampling.radial%i.coverage = options.mri.objectsize_in_pixel;', rad(i)))
    eval(sprintf('options.mri.sampling.radial%i.n_spokes = %i;', rad(i),rad(i)))
    eval(sprintf('options.mri.sampling.radial%i.center_block = center_block;', rad(i)))
    eval(sprintf('data.mri.freq.radial%i = MRI2_sample_radial_standard(options.mri.sampling.radial%i.coverage, [], options.mri.sampling.radial%i.n_spokes);',rad(i),rad(i),rad(i)))
    eval(sprintf('data.mri.freq.radial%i = [data.mri.freq.radial%i MRI2_sample_cartesian_x_standard(options.mri.sampling.radial%i.center_block)];',rad(i),rad(i),rad(i)))
end

% radial golden
rad = [16; 8; 4];
for i = 1 : size(rad,1)
    eval(sprintf('options.mri.sampling.radial_golden%i.coverage = options.mri.objectsize_in_pixel;', rad(i)))
    eval(sprintf('options.mri.sampling.radial_golden%i.n_spokes = %i;', rad(i),rad(i))) 
    eval(sprintf('options.mri.sampling.radial_golden%i.center_block = center_block;', rad(i)))
    eval(sprintf('data.mri.freq.radial_golden%i = MRI2_sample_radial_golden(options.mri.sampling.radial_golden%i.coverage, [], options.mri.sampling.radial_golden%i.n_spokes);',rad(i),rad(i),rad(i)))
    eval(sprintf('data.mri.freq.radial_golden%i = [data.mri.freq.radial_golden%i MRI2_sample_cartesian_x_standard(options.mri.sampling.radial_golden%i.center_block)];',rad(i),rad(i),rad(i)))
end

% radial random
rad = [16; 8; 4];
for i = 1 : size(rad,1)
    eval(sprintf('options.mri.sampling.radial_random%i.coverage = options.mri.objectsize_in_pixel;', rad(i)))
    eval(sprintf('options.mri.sampling.radial_random%i.n_spokes = %i;', rad(i),rad(i))) 
    eval(sprintf('options.mri.sampling.radial_random%i.center_block = center_block;', rad(i)))
    eval(sprintf('data.mri.freq.radial_random%i = MRI2_sample_radial_random(options.mri.sampling.radial_random%i.coverage, [], options.mri.sampling.radial_random%i.n_spokes);',rad(i),rad(i),rad(i)))
    eval(sprintf('data.mri.freq.radial_random%i = [data.mri.freq.radial_random%i MRI2_sample_cartesian_x_standard(options.mri.sampling.radial_random%i.center_block)];',rad(i),rad(i),rad(i)))
end

%% spiral
options.mri.sampling.spiral16.coverage = options.mri.objectsize_in_pixel;
options.mri.sampling.spiral16.n_spokes = 16;
options.mri.sampling.spiral16.n_turns = 1;
options.mri.sampling.spiral16.n_readouts = 3000;
options.mri.sampling.spiral16.uniformity = 2.5;
options.mri.sampling.spiral16.center_block = center_block;

options.mri.sampling.spiral8.coverage = options.mri.objectsize_in_pixel;
options.mri.sampling.spiral8.n_spokes = 8;
options.mri.sampling.spiral8.n_turns = 1;
options.mri.sampling.spiral8.n_readouts = 6000;
options.mri.sampling.spiral8.uniformity = 0.6;
options.mri.sampling.spiral8.center_block = center_block;

options.mri.sampling.spiral4.coverage = options.mri.objectsize_in_pixel;
options.mri.sampling.spiral4.n_spokes = 4;
options.mri.sampling.spiral4.n_turns = 2;
options.mri.sampling.spiral4.n_readouts = 5000;
options.mri.sampling.spiral4.uniformity = 1;
options.mri.sampling.spiral4.center_block = center_block;

options.mri.sampling.spiral2.coverage = options.mri.objectsize_in_pixel;
options.mri.sampling.spiral2.n_spokes = 2;
options.mri.sampling.spiral2.n_turns = 5;
options.mri.sampling.spiral2.n_readouts = 10000;
options.mri.sampling.spiral2.uniformity = 0.6;
options.mri.sampling.spiral2.center_block = center_block;

options.mri.sampling.spiralPhyll1.coverage = options.mri.objectsize_in_pixel;
options.mri.sampling.spiralPhyll1.density = .1;
options.mri.sampling.spiralPhyll1.center_block = center_block;

options.mri.sampling.spiralPhyll2.coverage = options.mri.objectsize_in_pixel;
options.mri.sampling.spiralPhyll2.density = .2;
options.mri.sampling.spiralPhyll2.center_block = center_block;

data.mri.freq.spiral16 = MRI2_sample_spiral_n(options.mri.sampling.spiral16.coverage, ...
    options.mri.sampling.spiral16.n_readouts, options.mri.sampling.spiral16.n_turns, ...
    options.mri.sampling.spiral16.uniformity, [], options.mri.sampling.spiral16.n_spokes);
data.mri.freq.spiral16 = [data.mri.freq.spiral16 MRI2_sample_cartesian_x_standard(options.mri.sampling.spiral16.center_block)];

data.mri.freq.spiral8 = MRI2_sample_spiral_n(options.mri.sampling.spiral8.coverage, ...
    options.mri.sampling.spiral8.n_readouts, options.mri.sampling.spiral8.n_turns, ...
    options.mri.sampling.spiral8.uniformity, [], options.mri.sampling.spiral8.n_spokes);
data.mri.freq.spiral8 = [data.mri.freq.spiral8 MRI2_sample_cartesian_x_standard(options.mri.sampling.spiral8.center_block)];

data.mri.freq.spiral4 = MRI2_sample_spiral_n(options.mri.sampling.spiral4.coverage, ...
    options.mri.sampling.spiral4.n_readouts, options.mri.sampling.spiral4.n_turns, ...
    options.mri.sampling.spiral4.uniformity,[], options.mri.sampling.spiral4.n_spokes);
data.mri.freq.spiral4 = [data.mri.freq.spiral4 MRI2_sample_cartesian_x_standard(options.mri.sampling.spiral4.center_block)];

data.mri.freq.spiral2 = MRI2_sample_spiral_n(options.mri.sampling.spiral2.coverage, ...
    options.mri.sampling.spiral2.n_readouts, options.mri.sampling.spiral2.n_turns, ...
    options.mri.sampling.spiral2.uniformity, [], options.mri.sampling.spiral2.n_spokes);
data.mri.freq.spiral2 = [data.mri.freq.spiral2 MRI2_sample_cartesian_x_standard(options.mri.sampling.spiral2.center_block)];

data.mri.freq.spiralPhyll1 = MRI2_sample_spiral_phyllotaxis(options.mri.sampling.spiralPhyll1.coverage, ...
    options.mri.sampling.spiralPhyll1.density);
data.mri.freq.spiralPhyll1 = [data.mri.freq.spiralPhyll1 MRI2_sample_cartesian_x_standard(options.mri.sampling.spiralPhyll1.center_block)];

data.mri.freq.spiralPhyll2 = MRI2_sample_spiral_phyllotaxis(options.mri.sampling.spiralPhyll2.coverage, ...
    options.mri.sampling.spiralPhyll2.density);
data.mri.freq.spiralPhyll2 = [data.mri.freq.spiralPhyll2 MRI2_sample_cartesian_x_standard(options.mri.sampling.spiralPhyll2.center_block)];

%% regrid
D = fieldnames(data.mri.freq);
for d = 1:length(D)
    eval(sprintf('data.mri.freq.%s = round(data.mri.freq.%s);', D{d}, D{d}));
end
% clear multiple samples
D = fieldnames(data.mri.freq);
s = options.mri.objectsize_in_pixel;
for d = 1 : length(D)
    eval(sprintf('data.mri.freq.%s = MRI_ind2freq(unique(MRI_freq2ind(data.mri.freq.%s, s)),s);', D{d}, D{d}));
end
%% create random variations
D = fieldnames(data.mri.freq);
s = options.mri.objectsize_in_pixel;
dd = randi(length(D),1,2);
for d = dd
    eval(sprintf('data.mri.freq.%s_randvar = max(min(data.mri.freq.%s + round(2*(rand(size(data.mri.freq.%s))-.5)), s(1)/2-1), -s(1)/2);', D{d}, D{d}, D{d}));
end
% clear multiple samples
D = fieldnames(data.mri.freq);
s = options.mri.objectsize_in_pixel;
for d = 1 : length(D)
    eval(sprintf('data.mri.freq.%s = MRI_ind2freq(unique(MRI_freq2ind(data.mri.freq.%s, s)),s);', D{d}, D{d}));
end
%% show frequencies
show_kspace = round(1.1*options.mri.objectsize_in_pixel);
D = fieldnames(data.mri.freq);
for d = 1 : length(D)
    eval(sprintf('x = data.mri.freq.%s;',D{d}))
    m = MRI_sample_backward(ones(size(x,2),1), x, show_kspace);
    C = 256;
    imwrite((C-1)*m,colormap(gray(C)),sprintf('%s/%s_sampling_%s.png', options.misc.folder.data, options.misc.name, D{d}));
end

%% make it noisy    
options.mri.noise.std_rel = 0.05;
options.mri.noise.std_abs_T1 = options.mri.noise.std_rel * norm(FT1) / sqrt(numel(FT1));
options.mri.noise.std_abs_T2 = options.mri.noise.std_rel * norm(FT2) / sqrt(numel(FT2));

D = fieldnames(data.mri.freq);
T = {'T1','T2'};
for t = 1:length(T)
    for d = 1:length(D)
        eval(sprintf('data.mri.%s.%s.data = MRI_sample_forward(data.mri.freq.%s, F%s);', T{t}, D{d}, D{d}, T{t}));
        eval(sprintf('data.mri.%s.%s.data_noisy = MRI_noise(data.mri.%s.%s.data, options.mri.noise.std_abs_%s);', T{t}, D{d}, T{t}, D{d}, T{t}));
    end
end
%%
l = [];
for d = 1 : length(D)
    eval(sprintf('l = [l numel(data.mri.T1.%s.data)];', D{d}));
end
l = l./max(l)*100;

clf; barh(l); title('amount of data'); axis tight;
set(gca, 'YTick', 1:length(D), 'YTickLabel', D, 'TickLabelInterpreter','none');

%%
saveas(gcf, [options.misc.folder.data '/' options.misc.name '_sampling_num.png']);
%%
save([options.misc.folder.data '/' options.misc.name '.mat'], 'data', 'groundtruth', 'options');
