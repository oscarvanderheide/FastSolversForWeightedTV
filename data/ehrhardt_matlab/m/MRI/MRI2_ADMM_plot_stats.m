function MRI2_ADMM_plot_stats(stats, offset)
% MRI2_ADMM_plot_stats
%   MRI2_ADMM_plot_stats(stats, offset) plots the stats of MRI reconstruction 
% with ADMM. The struct stats is the output of that algorithm. The offset
% allows the user to skip the first iterations for better visibility.
%
% Input:    
%   stats [struct]              
%       struct with statistical output of the ADMM algorithm
%
%   offset [int]       
%       skip the first iterations
%        
% See also:
%
% -------------------------------------------------------------------------
%   changes:
% 
% 2015-06-18 --------------------------------------------------------------
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
    
    if nargin < 2; offset = 0; end;

    s = stats;
    I = (offset+1):length(s.fun);
    clf;
%     subplot(211); plotyy(I,[abs(s.fun(I)) s.data_fit(I) s.prior(I)], I, [abs(s.constraint1(I)) abs(s.constraint2(I))], 'semilogy'); 
    subplot(211); plotyy(I,[abs(s.fun(I)) s.data_fit(I) s.prior(I)], I, [abs(s.constraint1(I)) abs(s.constraint2(I))], 'loglog'); 
    title('fun, data, prior / const1, const2'); legend('abs fun', 'data', 'prior', 'abs constraint1', 'abs constraint2');
    try
        subplot(212); plotyy(I,[s.res_dual(I) s.res_primal(I)],I,s.rho(I),'loglog'); 
    catch
        subplot(212); plot(I,[s.res_dual(I) s.res_primal(I)],I,s.rho(I)); 
    end
    title(sprintf('residuals / rho, time (%3.2fs)', s.time(end))); legend('dual residual', 'primal residual', 'rho');

end