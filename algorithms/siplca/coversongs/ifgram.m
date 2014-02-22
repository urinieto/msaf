function [F,D] = ifgram(X, N, W, H, SR)
% [F,D] = ifgram(X, N, W, H, SR)       Instantaneous frequency by phase deriv.
%    X is a 1-D signal.  Process with N-point FFTs applying a W-point 
%    window, stepping by H points; return (N/2)+1 channels with the 
%    instantaneous frequency (as a proportion of the sampling rate) 
%    obtained as the time-derivative of the phase of the complex spectrum
%    as described by Toshihiko Abe, Takao Kobayashi, and Satoshi Imai
%    "Robust Pitch Estimation with Harmonics Enhancement in Noisy 
%    Environments Based on Instantaneous Frequency" ICSLP 1996
%    http://www.kbys.ip.titech.ac.jp/research/pdf/icslp96-pitch.pdf
%    See also Abe's 2006 IEEE TASLP paper 14(4) 1292-1300.
% 
%    Same arguments and some common code as dpwebox/stft.m.
%    Calculates regular STFT as side effect - returned in D.
% after 1998may02 dpwe@icsi.berkeley.edu
% 2001-03-05 dpwe@ee.columbia.edu  revised version
% 2001-12-13 dpwe@ee.columbia.edu  Fixed to work when N != W
% $Header: $

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

if nargin < 2;  N = 256; end
if nargin < 3;  W = N;   end
if nargin < 4;  H = W/2; end
if nargin < 5;  SR = 1;  end

s = length(X);
% Make sure it's a single row
if size(X,1) > 1
  X = X';
end

%win = [0,hanning(W-1)'];
win = 0.5*(1-cos([0:(W-1)]/W*2*pi));

% Window for discrete differentiation
T = W/SR;
dwin = -pi / T * sin([0:(W-1)]/W*2*pi);

% sum(win) takes out integration due to window, 2 compensates for neg frq
norm = 2/sum(win);

% How many complete windows?
nhops = 1 + floor((s - W)/H);

F = zeros(1 + N/2, nhops);
D = zeros(1 + N/2, nhops);

nmw1 = floor( (N-W)/2 );
nmw2 = N-W - nmw1;

ww = 2*pi*[0:(N-1)]*SR/N;

for h = 1:nhops
  u = X((h-1)*H + [1:W]);
%  if(h==0)
%	plot(u)
%  end
  % Apply windows now, while the length is right
  wu = win.*u;
  du = dwin.*u;
  
  % Pad or truncate samples if N != W
  if N > W
    wu = [zeros(1,nmw1),wu,zeros(1,nmw2)];
    du = [zeros(1,nmw1),du,zeros(1,nmw2)];
  end
  if N < W
    wu = wu(-nmw1+[1:N]);
    du = du(-nmw1+[1:N]);
  end
  % FFTs of straight samples plus differential-weighted ones
  t1 = fft(fftshift(du));
  t2 = fft(fftshift(wu));
  % Scale down to factor out length & window effects
  D(:,h) = t2(1:(1 + N/2))'*norm;

  % Calculate instantaneous frequency from phase of differential spectrum
  t = t1 + j*(ww.*t2);
  a = real(t2);
  b = imag(t2);
  da = real(t);
  db = imag(t);
  instf = (1/(2*pi))*(a.*db - b.*da)./((a.*a + b.*b)+(abs(t2)==0));
  % 1/2pi converts rad/s into cycles/s
  % sampling rate already factored in as constant in dwin & ww
  % so result is in Hz
  
  F(:,h) = instf(1:(1 + N/2))';
    
end;

