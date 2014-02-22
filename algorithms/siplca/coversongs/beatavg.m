function X = beatavg(Y,bts)
% X = beatavg(Y,bys)
%    Calculate average of columns of Y according to grid defined 
%    (real-valued) column indices in vector bts.
%    For folding spectrograms down into beat-sync features.
% 2006-09-26 dpwe@ee.columbia.edu

% beat-based segments
%bts = beattrack(d,sr);
nbts = length(bts);
bttime = mean(diff(bts));
% map beats to specgram slices
ncols = size(Y,2);
coltimes = [0:(ncols-1)];
cols2beats = zeros(nbts, ncols);
btse = [bts,max(coltimes)];
for b = 1:nbts
  cols2beats(b,:) = ((coltimes >= btse(b)) & (coltimes < btse(b+1)))*1/(btse(b+1)-btse(b));
end

% The actual desired output
X = Y * cols2beats';
