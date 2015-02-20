function Y = chromrot(X,N)
% Y = chromrot(X,N)
%    Rotate each column of chroma feature matrix X down by N
%    semitones.
% 2006-07-15 dpwe@ee.columbia.edu

[nr,nc] = size(X);

Y = X(1+rem([0:(nr-1)]+N+nr,nr),:);
