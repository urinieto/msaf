function [So, Su] = eval_segmentation_entropy(estframelabs, gtframelabs)
% [So Su] = eval_segmentation_entropy(estframelabs, gtframelabs)
%
% Computes structural segmentation performance metrics proposed in:
% H. Lukashevich. "Towards Quantitative Measure of Evaluating Song
% Segmentation". In Proc. ISMIR, 2008.
%
% 2009-12-22 Ron Weiss <ronw@nyu.edu>

gtlabels = unique(gtframelabs);
estlabels = unique(estframelabs);

N = length(estframelabs);
Na = length(gtlabels);
Ne = length(estlabels);

nij = zeros(Na, Ne);
nia = zeros(1, Na);
nje = zeros(1, Ne);
for i = 1:Na
  curra = 1.0 * (gtframelabs == gtlabels(i)) + eps; 
  nia(i) = sum(curra);
  for j = 1:Ne
    curre = 1.0 * (estframelabs == estlabels(j));
    nij(i,j) = curra * curre' + eps;
  end
end
for j = 1:Ne
  curre = 1.0 * (estframelabs == estlabels(j)) + eps;
  nje(j) = sum(curre);
end

norm = sum(sum(nij));
pij = nij / norm;
pia = nia / norm;
pje = nje / norm;
pijae = nij ./ repmat(nje, [Na 1]);
pjiea = nij' ./ repmat(nia, [Ne 1]);

HEA = - sum(pia .* sum(pjiea .* log2(pjiea), 1));
HAE = - sum(pje .* sum(pijae .* log2(pijae), 1));

So = 1 - HEA / log2(Ne);
Su = 1 - HAE / log2(Na);

if isnan(So) || isinf(So)
  So = 0
end
if isnan(Su) || isinf(Su)
  Su = 0
end

