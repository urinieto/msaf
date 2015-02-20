function [p,m,S] = ifptrack(d,w,sr,fminl,fminu,fmaxl,fmaxu)
% [p,m,S] = ifptrack(d,w,sr,fminl,fminu,fmaxl,fmaxu)
%     Pitch track based on inst freq.
%     Look for adjacent bins with same inst freq.
%     d is the input waveform.  sr is its sample rate
%     w is the basic STFT DFT length (window is half, hop is 1/4)
%     S returns the underlying complex STFT.
%     fmin,fmax define ramps at edge of sensitivity
% 2006-05-03 dpwe@ee.columbia.edu

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

% downweight fundamentals below here
if nargin < 4; fminl = 150; end
if nargin < 5; fminu = 300; end
% highest frequency we look to
if nargin < 6; fmaxl = 2000; end
if nargin < 7; fmaxu = 4000; end


% Calculate the inst freq gram
[I,S] = ifgram(d,w,w/2,w/4,sr);

% Only look at bins up to 2 kHz
maxbin = round(fmaxu * (w/sr) );
%maxbin = size(I,1)
minbin = round(fminl * (w/sr) );

% Find plateaus in ifgram - stretches where delta IF is < thr
ddif = [I(2:maxbin, :);I(maxbin,:)] - [I(1,:);I(1:(maxbin-1),:)];

% expected increment per bin = sr/w, threshold at 3/4 that
dgood = abs(ddif) < .75*sr/w;

% delete any single bins (both above and below are zero);
dgood = dgood .* ([dgood(2:maxbin,:);dgood(maxbin,:)] >  0 | [dgood(1,:);dgood(1:(maxbin-1),:)] > 0);

% check it out
%p = dgood;

% reconstruct just pitchy cells?
%r = istft(p.*S,w,w/2,w/4);

p = 0*dgood;
m = 0*dgood;

% For each frame, extract all harmonic freqs & magnitudes
for t = 1:size(I,2)
  ds = dgood(:,t)';
  lds = length(ds);
  % find nonzero regions in this vector
  st = find(([0,ds(1:(lds-1))]==0) & (ds > 0));
  en = find((ds > 0) & ([ds(2:lds),0] == 0));
  npks = length(st);
  frqs = zeros(1,npks);
  mags = zeros(1,npks);
  for i = 1:length(st)
    bump = abs(S(st(i):en(i),t));
    frqs(i) = (bump'*I(st(i):en(i),t))/(sum(bump)+(sum(bump)==0));
    mags(i) = sum(bump);
    if frqs(i) > fmaxu
      mags(i) = 0;
      frqs(i) = 0;
    elseif frqs(i) > fmaxl
      mags(i) = mags(i) * max(0, (fmaxu - frqs(i))/(fmaxu-fmaxl));
    end
    % downweight magnitudes below? 200 Hz
    if frqs(i) < fminl
      mags(i) = 0;
      frqs(i) = 0;
    elseif frqs(i) < fminu
      % 1 octave fade-out
      mags(i) = mags(i) * (frqs(i) - fminl)/(fminu-fminl);
    end
    if frqs(i) < 0 
      mags(i) = 0;
      frqs(i) = 0;
    end
    
  end

% then just keep the largest at each frame (for now)
%  [v,ix] = max(mags);
%  p(t) = frqs(ix);
%  m(t) = mags(ix);
  % No, keep them all
  %bin = st;
  bin = round((st+en)/2);
  p(bin,t) = frqs;
  m(bin,t) = mags;
end

%% Pull out the max in each column
%[mm,ix] = max(m);
%% idiom to retrieve different element from each column
%[nr,nc] = size(p);
%pp = p((nr*[0:(nc-1)])+ix);
%mm = m((nr*[0:(nc-1)])+ix);
% r = synthtrax(pp,mm,sr,w/4);

%p = pp;
%m = mm;

