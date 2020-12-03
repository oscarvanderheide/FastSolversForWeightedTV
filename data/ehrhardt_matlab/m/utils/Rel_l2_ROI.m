% xxx.
%
% 2014-01-23 --------------------------------------------------------------
% 
% Matthias J. Ehrhardt
% CMIC, University College London, UK
% 
% matthias.ehrhardt.11@ucl.ac.uk
%
%   changes:
%       2014-01-23 xxx
% 
% -------------------------------------------------------------------------

function r = Rel_l2_ROI(x, x0, ROI)
    r = L2(x(ROI) - x0(ROI)) / (L2(x0(ROI)) + 1e-12);
end    