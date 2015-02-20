function Y = chromagram_IF(d,sr,fftlen,nbin,f_ctr,f_sd)
% Y = chromagram_IF(d,sr,fftlen,nbin,f_ctr,f_sd)
%  Calculate a "chromagram" of the sound in d (at sampling rate sr)
%  Use windows of fftlen points, hopped by ffthop points
%  Divide the octave into nbin steps
%  Weight with center frequency f_ctr (in Hz) and gaussian SD f_sd
%  (in octaves)
%  Use instantaneous frequency to keep only real harmonics.
% 2006-09-26 dpwe@ee.columbia.edu

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

if nargin < 3;   fftlen = 2048; end
if nargin < 4;   nbin = 12; end
if nargin < 5;   f_ctr = 1000; end
if nargin < 6;   f_sd = 1; end

A0 = 27.5; % Hz
A440 = 440; % Hz
f_ctr_log = log(f_ctr/A0) / log(2);

fminl = octs2hz(hz2octs(f_ctr)-2*f_sd);
fminu = octs2hz(hz2octs(f_ctr)-f_sd);
fmaxl = octs2hz(hz2octs(f_ctr)+f_sd);
fmaxu = octs2hz(hz2octs(f_ctr)+2*f_sd);

ffthop = fftlen/4;
nchr = 12;

% Calculate spectrogram and IF gram pitch tracks...
[p,m]=ifptrack(d,fftlen,sr,fminl,fminu,fmaxl,fmaxu); 

[nbins,ncols] = size(p);

%disp(['ncols = ',num2str(ncols)]);

% chroma-quantized IF sinusoids
Pocts = hz2octs(p+(p==0));
Pocts(p(:)==0) = 0;
% Figure best tuning alignment
nzp = find(p(:)>0);
%hist(nchr*Pmapo(nzp)-round(nchr*Pmapo(nzp)),100)
[hn,hx] = hist(nchr*Pocts(nzp)-round(nchr*Pocts(nzp)),100);
centsoff = hx(find(hn == max(hn)));
% Adjust tunings to align better with chroma
Pocts(nzp) = Pocts(nzp) - centsoff(1)/nchr;

% Quantize to chroma bins
PoctsQ = Pocts;
PoctsQ(nzp) = round(nchr*Pocts(nzp))/nchr;

% map IF pitches to chroma bins
Pmapc = round(nchr*(PoctsQ - floor(PoctsQ)));
Pmapc(p(:) == 0) = -1; 
Pmapc(Pmapc(:) == nchr) = 0;

Y = zeros(nchr,ncols);
for t = 1:ncols;
  Y(:,t)=(repmat([0:(nchr-1)]',1,size(Pmapc,1))==repmat(Pmapc(:,t)',nchr,1))*m(:,t);
end
