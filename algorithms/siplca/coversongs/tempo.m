function [t,xcr,D,onsetenv,sgsrate] = tempo(d,sr,tmean,tsd,onsetenv,debug)
% [t,xcr,D,onsetenv,sgsrate] = tempo(d,sr,tmean,tsd,onsetenv,debug)
%    Estimate the overall tempo of a track for the MIREX McKinney
%    contest.  
%    d is the input audio at sampling rate sr.  tmean is the mode
%    for BPM weighting (in bpm) and tsd is its spread (in octaves).
%    onsetenv is an already-calculated onset envelope (so d is
%    ignored).  debug causes a debugging plot.
%    Output t(1) is the lower BPM estimate, t(2) is the faster,
%    t(3) is the relative weight for t(1) compared to t(2).
%    xcr is the windowed autocorrelation from which the BPM peaks were picked.
%    D is the mel-freq spectrogram
%    onsetenv is the "onset strength waveform", used for beat tracking
%    sgsrate is the sampling rate of onsetenv and D.
%
% 2006-08-25 dpwe@ee.columbia.edu
% uses: localmax, fft2melmx

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

if nargin < 3;   tmean = 120; end
if nargin < 4;   tsd = 3.0; end
if nargin < 5;   onsetenv = []; end
if nargin < 6;   debug = 0; end

sro = 8000;
% specgram: 256 bin @ 8kHz = 32 ms / 4 ms hop
swin = 256;
shop = 32;
% mel channels
nmel = 40;
% sample rate for specgram frames (granularity for rest of processing)
sgsrate = sro/shop;
% autoco out to 4 s
acmax = round(4*sgsrate);

D = 0;
  
if length(onsetenv) == 0
  % no onsetenv provided - have to calculate it

  % resample to 8 kHz
  if (sr ~= sro)
    gg = gcd(sro,sr);
    d = resample(d,sro/gg,sr/gg);
    sr = sro;
  end

  D = specgram(d,swin,sr,swin,swin-shop);

  % Construct db-magnitude-mel-spectrogram
  mlmx = fft2melmx(swin,sr,nmel);
  D = 20*log10(max(1e-10,mlmx(:,1:(swin/2+1))*abs(D)));

  % Only look at the top 80 dB
  D = max(D, max(max(D))-80);

  % The raw onset decision waveform
  mm = (mean(max(0,diff(D')')));
  eelen = length(mm);

  % dc-removed mm
  onsetenv = filter([1 -1], [1 -.99],mm);

end  % of onsetenv calc block

% Find rough global period
% Only use the 1st 90 sec to estimate global pd (avoid glitches?)

maxdur = 90; % sec
maxcol = min(round(maxdur*sgsrate),length(onsetenv));

xcr = xcorr(onsetenv(1:maxcol),onsetenv(1:maxcol),acmax);

% find local max in the global ac
rawxcr = xcr(acmax+1+[0:acmax]);

% window it around default bpm
xcrwin = exp(-.5*((log((60*sgsrate./([0:acmax]+0.1)/tmean))/log(2)*tsd).^2));
xcr = rawxcr.*xcrwin;

xpks = localmax(xcr);  
% will not include any peaks in first down slope (before goes below
% zero for the first time)
xpks(1:min(find(xcr<0))) = 0;
% largest local max away from zero
maxpk = max(xcr(xpks));

% ?? then period is shortest period with a peak that approaches the max
%maxpkthr = 0.4;
%startpd = -1 + min(find( (xpks.*xcr) > maxpkthr*maxpk ) );
%startpd = -1 + (find( (xpks.*xcr) > maxpkthr*maxpk ) );

% no, just largest peak after windowing
startpd = -1 + find((xpks.*xcr) == max(xpks.*xcr));

% ??Choose acceptable peak closest to 120 bpm
%[vv,spix] = min(abs(60./(startpd/sgsrate) - 120));
%startpd = startpd(spix);
% No, just choose shortest acceptable peak
startpd = startpd(1);

t = 60/(startpd/sgsrate);

% Choose best peak out of .33 .5 2 3 x this period
candpds = round([.33 .5 2 3]*startpd);
candpds = candpds(candpds < acmax);

[vv,xx] = max(xcr(1+candpds));

startpd2 = candpds(xx);
vvm = xcr(1+startpd);
pratio = vvm/(vvm+vv);

t = [60/(startpd/sgsrate) 60/(startpd2/sgsrate) pratio];

% ensure results are lowest-first
if t(2) < t(1)
  t([1 2]) = t([2 1]);
  t(3) = 1-t(3);
end  

if debug > 0

  % Report results and plot weighted autocorrelation with picked peaks
  disp(['Global bt pd = ',num2str(t(1)),' @ ',num2str(t(3)),' / ',num2str(t(2)),' bpm']);

  subplot(414)
  plot([0:acmax],xcr,'-b', ...
       [0:acmax],xcrwin*maxpk,'-r', ...
       [startpd startpd], [min(xcr) max(xcr)], '-g', ...
       [startpd2 startpd2], [min(xcr) max(xcr)], '-c');
  grid;

end

% Read in all the tempo settings
% for i = 1:20; f = fopen(['mirex-beattrack/train/train',num2str(i),'-tempo.txt']); r(i,:) = fscanf(f, '%f\n'); fclose(f); end
