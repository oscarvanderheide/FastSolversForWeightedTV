function freq = MRI2_sample_spiral_phyllotaxis(s_kspace, density, phi_0)
% MRI2_sample_spiral_phyllotaxis
%   freq = MRI2_sample_spiral_phyllotaxis(s_kspace, density, phi_0) samples
% along the spiral phyllotaxis [1].
%
% ]1] Piccini, D., Littmann, A., Nielles-Vallespin, S., & Zenge, M. O. (2011). 
% Spiral phyllotaxis: the natural way to construct a 3D radial trajectory in MRI. 
% Magnetic Resonance in Medicine, 66(4), 1049–56. doi:10.1002/mrm.22898
% [2] http://blog.wolfram.com/2011/07/28/how-i-made-wine-glasses-from-sunflowers/
% [3] Vogel, H. (1979). A Better Way to Construct the Sunflower Head. Mathematical Biosciences, 44, 179-189.
%
% Input:
%   s_kspace [vector; DEFAULT = [128 128]]
%       size of the k-space in which we want to sample.
%
%   density [float; DEFAULT = .1]
%       density of the spiral
%
%   phi_0 [float; DEFAULT = 0]
%       first angle
%
% Output:
%   freq [matrix]
%       matrix where each column are the frequencies of a data point in
%       kspace.
%
% See also:
%
% -------------------------------------------------------------------------
%   changes:
% 
% 2015-07-02 --------------------------------------------------------------
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

    if nargin < 1; s_kspace = [128 128]; end;    
    if nargin < 2; density = .1; end;    
    if nargin < 3; phi_0 = 0; end;
    
    n_readouts = prod(density*(s_kspace-2));
    scale = 1./(2*density);
       
    n = 0:n_readouts-1;
    phi_gold = 2.4;
   
    phi = phi_gold*n + phi_0;
    r = scale*sqrt(n);
    
    freq = [r.*cos(phi); r.*sin(phi)];
    
end