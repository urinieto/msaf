function Y = chrompwr(X,P)
% Y = chrompwr(X,P)  raise chroma columns to a power, preserving norm
% 2006-07-12 dpwe@ee.columbia.edu

[nbins,nframes] = size(X);

% norms of each input col
CMn = repmat(sqrt(sum(X.^2)),nbins,1);
CMn(CMn == 0) = 1;

% normalize each input col, raise to power
CMp = (X./CMn).^P;

% norms of each resultant column
CMpn = repmat(sqrt(sum(CMp.^2)),nbins,1);
CMpn(CMpn == 0) = 1;

% rescale cols so norm of output cols match norms of input cols
Y = CMn.*(CMp./CMpn);

