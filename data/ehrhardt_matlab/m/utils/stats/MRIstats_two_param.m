function [min_sol, min_i, min_j] = MRIstats_two_param(solutions, groundtruth, param, param_str, filename)
% MRIstats_two_param
%   [min_sol, min_i, min_j] = MRIstats_one_param(solutions, groundtruth, param, param_str, filename)
% computes statistics and optimal solutions from an array of solutions.
%
% Input:    
%   solutions [cell]              
%       array of solutions
%
%   groundtruth [matrix]              
%       desired image
%
%   param [vector]              
%       vector of parameters that have been tested
%
%   param_str [string]              
%       name of these parameters
%
%   filename [string]              
%       name of the data set
%
% Output:
%   min_sol [matrix]
%       best solution
%
%   min_i [int]
%       which row index corresponds to the best solution
%
%   min_j [int]
%       which column index corresponds to the best solution
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
    
    save_images = nargin > 4;
    
    N = length(param{1});
    M = length(param{2});
    e = nan(N,M,2);
    
    for n = 1 : N
        for m = 1 : M
            if numel(solutions{n,m}) > 0
                e(n,m,1) = psnr(solutions{n,m},groundtruth);
                e(n,m,2) = 100*ssim(solutions{n,m},groundtruth);
            end
        end
    end

    [ee, min_i] = max(e(:,:,2));
    [~, min_j] = max(ee);
    min_i = min_i(min_j);

    for k = 1 : size(e,3)
        figure(1); clf;
        i = 1 : length(param{1});
        j = 1 : length(param{2});
        imagesc(i,j,e(i,j,k)'); colorbar; colormap parula; axis image;
        set(gca, 'XTick', i, 'YTick', j, 'XTickLabel', sprintf('%3.1e\n', param{1}(i)), 'YTickLabel', sprintf('%3.1e\n', param{2}(j)));
        xlabel(param_str{1}); ylabel(param_str{2});
        switch k
            case 1
                lab = 'PSNR';
            case 2
                lab = 'SSIM';
        end
        title(sprintf('opt %s:%3.1e, opt %s:%3.1e, %s@opt:%2.1f', param_str{1}, param{1}(min_i), param_str{2}, param{2}(min_j), lab, e(min_i,min_j,k)))
        if save_images; saveas(gcf, [filename '_' lab '.png']); end;
    end
    
    figure(2); clf;
    con_sol = MRIstats_concat_solutions(solutions);
    imagesc(con_sol); axis image; colorbar;
    set(gca, 'XTick', (i-.5)*size(solutions{1},1), 'YTick', (j-.5)*size(solutions{1},2), 'XTickLabel', sprintf('%3.1e\n', param{1}(i)), 'YTickLabel', sprintf('%3.1e\n', param{2}(j)));
    xlabel(param_str{1}); ylabel(param_str{2});
    colormap(gray(1024));
    if save_images;        
        C = 256;
        imwrite((C-1)*con_sol,colormap(gray(C)), [filename '_some_solutions_image.png']);
        saveas(gcf, [filename '_some_solutions.png']);
        set(gca, 'clim', [0,1])
        saveas(gcf, [filename '_some_solutions_thresh.png']); 
    end
    
    min_sol = solutions{min_i,min_j};
    if save_images
        C = 256;
        cmp = colormap(gray(C));
        imwrite((C-1)*min_sol, cmp, [filename '_best_solution.png']);

        err_sol = (min_sol - groundtruth);
        param = .4;
        cmp_e = colormap(redwhiteblue(C));
        imwrite((C-1) * (err_sol/(2*param) + .5), cmp_e, [filename '_best_error.png']);
        
        [~, ssim_map] = ssim(min_sol, groundtruth);
        imwrite((C-1) * ssim_map.^2, cmp, [filename '_best_error_SSIM.png']);
    end
end