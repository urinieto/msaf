function [F,bts] = chrombeatftrs(d,sr,f_ctr,f_sd,type)
% [F,bts] = chrombeatftrs(D,SR,F_CTR,F_SD,TYPE)
%    F returns a feature vector of beat-level chroma features (12
%    rows x n time step columns). bts returns the times of all the
%    beats.  
%    New version separates out chroma calculation
%    TYPE selects chroma calculation type; 1 (default) uses IF; 
%    2 uses all FFT bins, 3 uses only local peaks (a bit like Emilia).
% 2006-07-14 dpwe@ee.columbia.edu

%   Copyright (c) 2006 Columbia University.
% 
%   This file is part of LabROSA-coversongID
% 
%   LabROSA-coversongID is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License version 2 as
%   published by the Free Software Foundation.
% 
%   LabROSA-coversongID is distributed in the hope that it will be useful, but
%   WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%   General Public License for more details.
% 
%   You should have received a copy of the GNU General Public License
%   along with LabROSA-coversongID; if not, write to the Free Software
%   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
%   02110-1301 USA
% 
%   See the file "COPYING" for the text of the license.

if nargin < 3; f_ctr = 1000; end
if nargin < 4; f_sd = 1; end
if nargin < 5; type = 1; end


tempomean = 240;
%temposd = 1.5;
% Following the temposd setting from
% http://labrosa.ee.columbia.edu/projects/coversongs/covers80/
% to make the output of this function consistent with the features used
% in my ISMIR paper. -ronw 2010-06-17
temposd = 1.0;

% Try beat tracking now for quick answer
bts = beat(d,sr,[tempomean temposd],[6 0.8],0);

% Calculate frame-rate chromagram
fftlen = 2 ^ (round(log(sr*(2048/22050))/log(2)));
nbin = 12;

if type == 2
  Y = chromagram_E(d,sr,fftlen,nbin,f_ctr,f_sd);
elseif type == 3
  Y = chromagram_P(d,sr,fftlen,nbin,f_ctr,f_sd);
else
  Y = chromagram_IF(d,sr,fftlen,nbin,f_ctr,f_sd);
end  
  
ffthop = fftlen/4;
sgsrate = sr/ffthop;

F = beatavg(Y,bts*sgsrate);
