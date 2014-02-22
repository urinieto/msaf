function [b,onsetenv,D,cumscore] = beat(d,sr,startbpm,tightness,doplot)
% [b,onsetenv,D,cumscore] = beat(d,sr,startbpm,tightness,doplot)
%   b returns the times (in sec) of the beats in the waveform d, samplerate sr.
%   startbpm specifies the target tempo.  If it is a two-element
%   vector, it is taken as the mode of a tempo search window, with 
%   the second envelope being the spread (in octaves) of the
%   search, and the best tempo is calculated (with tempo.m).
%   tightness controls how tightly the start tempo is enforced
%   within the beat (default 6, larger = more rigid); if it is a 
%   two-element vector the second parameter is alpha, the strength 
%   of transition costs relative to local match (0..1, default 0.7).
%   doplot enables diagnostic plots; if it has two elements, they
%   are the time range (in sec) for the diagnostic plots.
%   onsetenv returns the raw onset detection envelope
%   D returns the mel-spectrogram, 
%   cumscore returns the per-frame cumulated dynamic-programming score.
% 2006-08-25 dpwe@ee.columbia.edu
% uses: localmax

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

if nargin < 3;   startbpm = 0; end
if nargin < 4;   tightness = 0; end
if nargin < 5;   doplot = 0; end

if length(startbpm) == 2
  temposd = startbpm(2);
  startbpm = startbpm(1);
else
  temposd = 0; 
end
if length(tightness) == 2
  alpha = tightness(2);
  tightness = tightness(1);
else
  alpha = 0.8
end
if tightness == 0;  tightness = 6; end

% Have we been given an envelope (nonnegative waveform)
if min(d) >= 0
  onsetenv = d;
  sgsrate = sr;
  disp(['beat: treating input as onset strength envelope']);
else
  onsetenv = [];
end

% debug/plotting options
plotlims = [];
if length(doplot) > 1
  % specify zoom-in limits too
  plotlims = doplot;
  doplot = 1;
end
if doplot > 0;  debug = 1; else debug = 0; end

b = [];

% Select tempo search either with startbpm = 0 (means use defaults)
% or startbpm > 0 but temposd > 0 too (means search around startbpm)
% If onsetenv is empty, have to run tempo too to convert waveform
% to onsetenv, but we might not use the tempo it picks.
if startbpm == 0 | temposd > 0 | length(onsetenv) == 0

  if startbpm == 0
    tempomean = 120;
  else
    tempomean = startbpm;
  end

  if temposd == 0
    temposd = 0.7;
  end
  
  % Subfunction estimates global BPM; returns 'onset strength'
  % waveform onsetenv
  % If we were given an onsetenv as input, will use that
  [t,xcr,D,onsetenv,sgsrate] = tempo(d,sr,tempomean,temposd,onsetenv,debug);
  
  % tempo.m returns the top-2 BPM estimates; use faster one for
  % beat tracking
  if (startbpm == 0 | temposd > 0)
    startbpm = max(t([1 2]));
  end

  if debug == 1
    % plot the mel-specgram
    tt = [1:length(onsetenv)]/sgsrate;
    subplot(411)
    imagesc(tt,[1 40],D); axis xy
    subplot(412)
    plot(tt,onsetenv);
  end

end

% convert startbpm to startpd
startpd = (60*sgsrate)/startbpm;
%disp(['startpd=',num2str(startpd)]);

pd = startpd;
  
% Smooth beat events
templt = exp(-0.5*(([-pd:pd]/(pd/32)).^2));
localscore = conv(templt,onsetenv);
localscore = localscore(round(length(templt)/2)+[1:length(onsetenv)]);

% DP version:
% backlink(time) is index of best preceding time for this point
% cumscore(time) is total cumulated score to this point

backlink = zeros(1,length(localscore));
cumscore = zeros(1,length(localscore));

% search range for previous beat
prange = round(-2*pd):-round(pd/2);

% Skewed window
txwt = exp(-0.5*((tightness*log(prange/-pd)).^2));

starting = 1;
for i = 1:length(localscore)
  
  timerange = i + prange;
  
  % Are we reaching back before time zero?
  zpad = max(0, min(1-timerange(1),length(prange)));

  % Search over all possible predecessors and apply transition 
  % weighting
  scorecands = txwt .* [zeros(1,zpad),cumscore(timerange(zpad+1:end))];
  % Find best predecessor beat
  [vv,xx] = max(scorecands);
  % Add on local score
  cumscore(i) = alpha*vv + (1-alpha)*localscore(i);

  % special case to catch first onset
%  if starting == 1 & localscore(i) > 100*abs(vv)
  if starting == 1 & localscore(i) < 0.01*max(localscore);
    backlink(i) = -1;
  else
    backlink(i) = timerange(xx);
    % prevent it from resetting, even through a stretch of silence
    starting = 0;
  end
  
end

%%%% Backtrace

% Cumulated score is stabilized to lie in constant range, 
% so just look for one near the end that has a reasonable score
medscore = median(cumscore(localmax(cumscore)));
bestendx = max(find(cumscore .* localmax(cumscore) > 0.5*medscore));

b = bestendx;

while backlink(b(end)) > 0
  b = [b,backlink(b(end))];
end

b = fliplr(b);

% return beat times in secs
b = b / sgsrate;

% Debug visualization
if doplot == 1
  subplot(411)
  hold on;
  plot([b;b],[0;40]*ones(1,length(b)),'w');
  hold off;
  subplot(412)
  hold on;
  plot([b;b],[-5;20]*ones(1,length(b)),'g');
  hold off;

  % redo 3rd pane as xcorr with templt
  subplot(413)
  tt = [1:length(localscore)]/sgsrate;
  plot(tt,localscore);
  hold on; plot([b;b],[min(localscore);max(localscore)]*ones(1,length(b)),'g'); hold off
   
  if length(plotlims) > 0
    for i = 1:3;
      subplot(4,1,i)
      ax = axis;
      ax([1 2]) = plotlims;
      axis(ax);
    end
  end
  
end
