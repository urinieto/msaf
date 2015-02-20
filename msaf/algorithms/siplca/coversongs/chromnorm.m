function [N,S] = chromnorm(F)
% [N,S] = chromnorm(F)
%    Normalize each column of a chroma ftrvec to unit norm
%    so cross-correlation will give cosine distance
%    S returns the per-column original norms, for reconstruction
% 2006-07-14 dpwe@ee.columbia.edu

[nchr, nbts] = size(F);

S = sqrt(sum(F.^2));

N = F./repmat(S+(S==0), nchr, 1);
