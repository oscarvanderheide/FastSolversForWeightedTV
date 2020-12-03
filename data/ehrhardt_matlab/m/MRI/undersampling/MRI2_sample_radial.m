function freq = MRI2_sample_radial(s_kspace, n_readouts, i_angles)
% MRI2_sample_radial
%   freq = MRI2_sample_radial(s_kspace, n_samples_per_angle, i_angles) samples 
% the kspace radially.
%
% Input:
%   s_kspace [vector]
%       size of the k-space in which we want to sample.
%
%   n_readouts [int]
%       number of readouts in the frequency encoding direction
%
%   i_angles [vector]
%       angles for the spokes
%
% Output:
%   freq [matrix]
%       matrix where each column are the frequencies of a data point in
%       kspace.
%
% See also: MRI_sample_line
%
% -------------------------------------------------------------------------
%   changes:
% 
% 2014-08-06 --------------------------------------------------------------
% Matthias J. Ehrhardt
% CMIC, University College London, UK 
% matthias.ehrhardt.11@ucl.ac.uk
% http://www.cs.ucl.ac.uk/staff/ehrhardt/software.html
%
% -------------------------------------------------------------------------
% Copyright 2014 University College London
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

    if nargin < 1; s_kspace = [128 128]; end;    
    if nargin < 2 || numel(n_readouts) == 0; n_readouts = s_kspace(1)-(1-mod(s_kspace(1),2)); end;    
    if nargin < 3 || numel(i_angles) == 0; 
        n_angles = 10;
        i_angles = linspace(0, (n_angles-1)./n_angles * pi, n_angles); 
    end;
        
    if numel(i_angles) == 1 
        n_angles = i_angles;
        i_angles = linspace(0, (n_angles-1)./n_angles * pi, n_angles); 
    end;
    
    n_angles = length(i_angles);
    
    if numel(n_readouts) > 0
        freq = zeros(2,n_angles*n_readouts);
                
        r = ceil(-(s_kspace-1)/2);

        for n = 1 : n_angles
            phi = i_angles(n);

            i_start = r .* [cos(phi), sin(phi)];
            i_end   = -i_start;

            freq(:, n_readouts*(n-1)+1 : n_readouts*n)  = MRI_sample_line(i_start, i_end, n_readouts);
        end
    else
        freq = zeros(2,n_angles*n_readouts);
        r = ceil(-(s_kspace-1)/2);
        freq_used = 0;
        
        for n = 1 : n_angles
            phi = i_angles(n);

            i_start = r .* [cos(phi), sin(phi)];
            i_end   = -i_start;

            n_readouts = round(norm(i_start-i_end)) + 1;
            freq(:, (freq_used+1):(freq_used + n_readouts)) = MRI_sample_line(i_start, i_end, n_readouts);
            freq_used = freq_used + n_readouts;
        end
        
        freq(:,freq_used+1:end) = [];
    end
        
end