function out = Image_complex(x, model, brightness_max, brightness_thresh, period_phase, phase_shift)
% Image_complex
%   out = Image_complex(x, model, brightness_max, brightness_thresh, period_phase, phase_shift)
% computes a colored image that shows a complex image where the magnitude
% determines the brightness and the phase determines the colours.
%
% Input:    
%   x [matrix]              
%       complex image
%
%   model [int; DEFAULT = 1]              
%       the image model. There are currently two models implemented, cf.
%       http://www.mathematica-journal.com/issue/v7i2/articles/contents/thaller/html/
%
%   brightness_max [float; DEFAULT = 1]              
%       maximal brightness. This defines full brightness for model 1 and
%       white colour for model 2
%
%   brightness_thresh [float; DEFAULT = .5]              
%       only used in model 2. The threshold when the brightness behaviour
%       changes.
%
%   period_phase [float; DEFAULT = 2*pi]              
%       period of the phase
%
%   phase_shift [float; DEFAULT = 0]              
%       determines which colour corresponds to zero phase
%
% Output:
%   out [matrix; OPTIONAL]
%       colour image. If nargout = 0, then the image "out" is plotted with
%       the function "image".
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
        
    if nargin < 2 || isempty(model);                model = 1;              end;
    if nargin < 3 || isempty(brightness_max);       brightness_max = 1;     end;
    if nargin < 4 || isempty(brightness_thresh);    brightness_thresh = .5; end;
    if nargin < 5 || isempty(period_phase);         period_phase = 2*pi;    end;
    if nargin < 6 || isempty(phase_shift);          phase_shift = 0;        end;

    n_colours = 256;
    size_x = size(x);
    
    magn_x = abs(x);
    phas_x = Phas(x) + pi + phase_shift;
    phas_x = mod(phas_x, period_phase);
    
    colour_map = colormap(hsv(n_colours));
    
    phas_x = floor( phas_x(:) / period_phase * (n_colours-1) ) + 1;
    
    colour_x = colour_map(phas_x,:);
    
    magn_x = magn_x(:);
        
    brightness = min(magn_x, brightness_max) / brightness_max;
          
    colour_xx = zeros([size(x), 3]);
    
    for i = 1 : 3
        switch model
            case 1
                colour_xx(:,:,i) = reshape(colour_x(:,i) .* brightness, size_x);
            case 2
                ii = brightness <= brightness_thresh;
                colour_x(:,i) = ii .* colour_x(:,i) .* brightness / brightness_thresh ...
                    + (1 - ii) .* (colour_x(:,i) + (brightness - brightness_thresh)/(brightness_max - brightness_thresh) .* (1 - colour_x(:,i)));
                
                colour_xx(:,:,i) = reshape(colour_x(:,i), size_x);
        end
    end
    
    if nargout == 0
        image(colour_xx);
    else
        out = colour_xx;
    end
    
end