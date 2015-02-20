function [R,S,T,C] = coverTestLists(qlist,tlist,pwr,metric,verb);
% [R,S,T,C] = coverTestLists(qlist,tlist,pwr,norm,metric,verb);
%    Takes a list of query files qlist and compares each one to all 
%    of a list of test files tlist.
%    R returns a matrix of score values, each row is a query, each
%    column is one of the test elements.
%    S is a local normalization index; T is the best alignment time skew.
%    pwr is power to raise chroma vectors to (dflt 1).
%    metric is the metric used (1 = peak xcorr, 2 = peak filtered xcorr)
%    verb > 0 means provide progress update
% 2006-07-27 dpwe@ee.columbia.edu

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

if nargin < 2;   tlist = []; end
if nargin < 3;   pwr = 0.5;  end
if nargin < 4;   metric = 2; end
if nargin < 5;   verb = 0; end

nqlist = length(qlist);
ntlist = length(tlist);
if ntlist == 0
  tlist = qlist;
  ntlist = nqlist;
end

% Now run through the queries

for q = 1:nqlist
  
  qline = qlist{q};
  
  if (verb > 0)
    disp([datestr(rem(now,1),'HH:MM:SS'), ' ', 'doing song ',num2str(q),' ', qline]);
  end
  
  Q = load(qline);
  Q.F = chromnorm(chrompwr(Q.F,pwr));

  maxlag = 800;
  
  for i = 1:ntlist
    %    if (verb > 0)
    %      disp(['..versus ', tlist{i}]);
    %    end
    
    P = load(tlist{i});
    P.F = chromnorm(chrompwr(P.F,pwr));
    
    % Perform the cross-correlation of the two chroma beat ftr matrices
    r = chromxcorr(Q.F, P.F, maxlag);
    
    % find best alignments
    mmr = max(max(r));
    bestchrom = find(max(r') == mmr);
    
    if metric == 1
      R(q,i) = mmr;
      besttime = find(max(r) == mmr);
      S(q,i) = mean(mean(r(:,max(besttime-100,1):min(besttime+100,size(r,2)))));
      
    elseif metric == 2

      % Look for rapid variation - do HPF along time of best chrom
      fxc = filter([1 -1], [1 -.9], r(bestchrom,:)-mean(r(bestchrom,:)));
      % chop off first bit - onset transient for
      % start-in-the-middle
      fxc(1:50) = min(fxc);
      R(q,i) = max(fxc);
      refpt = maxlag;
    end
    besttime = find(fxc == max(fxc))-refpt-1;
    T(q,i) = besttime;
    C(q,i) = bestchrom;
    if verb > 0
      disp([datestr(rem(now,1),'HH:MM:SS'), ' ..versus ', tlist{i},' ',num2str(max(fxc)),' @ ',num2str(besttime)])
    end
      
    S(q,i) = sqrt(mean(fxc(max(besttime+refpt-100,1):min(besttime+refpt+100,length(fxc))).^2));
  end
end
