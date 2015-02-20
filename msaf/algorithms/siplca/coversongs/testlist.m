function [R,S,T] = testlist(queryflist,testflist,pwr,norm,metric,xcr,verb);
% [R,S,T] = testlist(queryflist,testflist,pwr,norm,metric,xcr,verb);
%    Takes a list of query files and compares each one to all 
%    of a list of test files.
%    R returns a matrix of score values, each row is a query, each
%    column is one of the test elements.
%    S is a local normalization index; T is the best alignment time skew.
%    pwr is power to raise chroma vectors to (dflt 1).
%    norm is the normalization mode for the xcorr (0 = none, 1 = by shorter)
%    metric is the metric used (1 = peak xcorr, 2 = peak filtered xcorr)
%    xcr = 1 means fast cross-correlation (default 0)
%    verb > 0 means provide progress update
% 2006-07-27 dpwe@ee.columbia.edu

if nargin < 3;   pwr = 1;  end
if nargin < 4;   norm = 1; end
if nargin < 5;   metric = 1; end
if nargin < 6;   xcr = 0; end
if nargin < 7;   verb = 0; end

[tsongs,ntsongs] = listfileread(testflist);
[qsongs,nqsongs] = listfileread(queryflist);

% Now run through the queries

for q = 1:nqsongs

  qline = qsongs{q};
  
  if (verb > 0)
    disp([datestr(rem(now,1),'HH:MM:SS'), ' ', 'doing song ',num2str(q),' ', qline]);
  end
  
  Q = load(qline);
  Q.F = chromnorm(chromapwr(Q.F,pwr));

  maxlag = 800;
  
  for i = 1:ntsongs
%    if (verb > 0)
%      disp(['..versus ', tsongs{i}]);
%    end
    
    P = load(tsongs{i});
    P.F = chromnorm(chromapwr(P.F,pwr));
    
    if xcr == 0
      r = chromxcorr2(Q.F, P.F, norm);
    else
      % fast version of xcor
      r = chromxcorr2fast(Q.F, P.F, norm, maxlag);
    end
    
    mmr = max(max(r));
    bestchrom = find(max(r') == mmr);
    besttime = find(max(r) == mmr);
    
    if metric == 1
      R(q,i) = mmr;
      S(q,i) = mean(mean(r(:,max(besttime-100,1):min(besttime+100,size(r,2)))));

    elseif metric == 2

      % Look for rapid variation - do HPF along time of best chrom
      fxc = filter([1 -1], [1 -.9], r(bestchrom,:)-mean(r(bestchrom,:)));
      % chop off first bit - onset transient for
      % start-in-the-middle
      fxc(1:50) = min(fxc);
      %
      R(q,i) = max(fxc);
      if (xcr == 0)
        refpt = length(Q.F);
      else
        refpt = maxlag;
      end
      besttime = find(fxc == max(fxc))-refpt;
      T(q,i) = besttime;
      if verb > 0
        disp([datestr(rem(now,1),'HH:MM:SS'), ' ..versus ', tsongs{i},' ',num2str(max(fxc)),' @ ',num2str(besttime)])
      end
      
      S(q,i) = sqrt(mean(fxc(max(besttime+refpt-100,1):min(besttime+refpt+100,length(fxc))).^2));
    end
  end
end

if (0)  % not wanted for submission version - self included in lists
  % scoring - find largest entry in each row (query)
  [vv,xx] = max(R');
  tt = 1:length(xx);
  aa = (xx == tt);
  disp(['Simple max: acc = ',num2str(mean(aa),3),' ',num2str(aa)]);
  [vv,xx] = max((R./S)');
  aa = (xx == tt);
  disp(['Ratio  max: acc = ',num2str(mean(aa),3),' ',num2str(aa)]);
end
