% MRI_run_recon
%   This script runs several reconstruction procedures
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

if run_without_prior
    %% without prior
    k = 1;
    timer = tic;
    for c = 1 : length(array_contrast)
        contrast = array_contrast{c};
        for d = 1 : length(array_data)
            sampling = array_data{d};
            
            fprintf('Progress: ')
            
            MRIrecon_without_prior(contrast, sampling, data, groundtruth, options, [], subfolder);
            
            fprintf('%d%%, %3.2fs\n', round(100*k./(length(array_contrast) * length(array_data))), toc(timer))
            k = k + 1;
        end
    end
end

if run_TV
    %% TV
    k = 1;
    timer = tic;
    for c = 1 : length(array_contrast)
        contrast = array_contrast{c};
        for d = 1 : length(array_data)
            sampling = array_data{d};
            
            fprintf('Progress: ')
            
            MRIrecon_TV(contrast, sampling, data, groundtruth, options, [], alphas, subfolder)
            
            fprintf('%d%%, %3.2fs\n', round(100*k./(length(array_contrast) * length(array_data))), toc(timer))
            k = k + 1;
        end
    end
end

if run_dTV
    %% AQPL
    k = 1;
    timer = tic;
    for c = 1 : length(array_contrast)
        contrast = array_contrast{c};
        for d = 1 : length(array_data)
            sampling = array_data{d};
            
            switch contrast
                case 'T1'
                    sideinfo = groundtruth.T2;
                    sideinfo_str = 'T2';
                    
                case 'T2'
                    sideinfo = groundtruth.T1;
                    sideinfo_str = 'T1';
            end
            
            fprintf('Progress: ')
            
            MRIrecon_dTV(contrast, sampling, '', sideinfo_str, sideinfo, data, groundtruth, options, [], alphas, etas, subfolder)
            
            fprintf('%d%%, %3.2fs\n', round(100*k./(length(array_contrast) * length(array_data))), toc(timer))
            k = k + 1;
        end
    end
end

if run_wTV
    %% wTV
    k = 1;
    timer = tic;
    for c = 1 : length(array_contrast)
        contrast = array_contrast{c};
        for d = 1 : length(array_data)
            sampling = array_data{d};
            
            switch contrast
                case 'T1'
                    sideinfo = groundtruth.T2;
                    sideinfo_str = 'T2';
                    
                case 'T2'
                    sideinfo = groundtruth.T1;
                    sideinfo_str = 'T1';
            end
            
            fprintf('Progress: ')
            
            MRIrecon_wTV(contrast, sampling, '', sideinfo_str, sideinfo, data, groundtruth, options, [], alphas, etas, subfolder)
            
            fprintf('%d%%, %3.2fs\n', round(100*k./(length(array_contrast) * length(array_data))), toc(timer))
            k = k + 1;
        end
    end
end