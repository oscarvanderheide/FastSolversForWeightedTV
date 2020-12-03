function [min_sol, min_i] = MRIstats_one_param(solutions, groundtruth, param, param_str, filename)
% MRIstats_one_param
%   [min_sol, min_i] = MRIstats_one_param(solutions, groundtruth, param, param_str, filename)
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
%       which index corresponds to the best solution
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
    
    N = length(solutions);
    e = nan(N,3);
    
    for n = 1 : N
        if numel(solutions{n}) > 0
            e(n,1) = psnr(solutions{n},groundtruth);
            e(n,2) = 100*ssim(solutions{n},groundtruth);
        end
    end

    figure(1); clf;
    i = find(~isnan(e(:,1)));
    plotyy(i,[e(i,1)],i,e(i,2));
    set(gca, 'XTick', i, 'XTickLabel', sprintf('%3.1e\n', param(i)));    
    xlabel(param_str); legend('PSNR [dB]', 'SSIM [%]');
    if save_images; saveas(gcf, [filename '_errors.png']); end;
    
    figure(2); clf;
    [~, min_i] = max(e(:,2));
    con_sol = MRIstats_plot_solution_array(solutions, e(:,1), min_i, param, param_str);
    colormap(gray(1024));
    if save_images;
        C = 256;
        imwrite((C-1)*con_sol,colormap(gray(C)), [filename '_some_solutions_image.png']);
        saveas(gcf, [filename '_some_solutions.png']);
        set(gca, 'clim', [0,1])
        saveas(gcf, [filename '_some_solutions_thresh.png']);
    end

    min_sol = solutions{min_i};
    if save_images
        C = 256;
        cmp = colormap(gray(C));
        imwrite((C-1)*min_sol, cmp, [filename '_best_solution.png']);

        err_sol = (min_sol - groundtruth);
        p = .4;
        cmp_e = colormap(redwhiteblue(C));
        imwrite((C-1) * (err_sol/(2*p) + .5), cmp_e, [filename '_best_error.png']);
        
        [~, ssim_map] = ssim(min_sol, groundtruth);
        imwrite((C-1) * ssim_map.^2, cmp, [filename '_best_error_SSIM.png']);
    end
end
