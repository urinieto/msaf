function [pfm, ppr, prr] = eval_segmentation_pairwise(estframelabs, gtframelabs)
% [pfm ppr prr] = eval_segmentation_pairwise(estframelabs, gtframelabs)
%
% Computes pairwise structural segmentation performance metrics
% described in:
% M. Levy and M. Sandler. Structural Segmentation of Musical Audio by
% Constrained Clustering. IEEE Trans.  Audio, Speech, and Language
% Processing, 16(2), 2008.
%
% 2009-12-22 Ron Weiss <ronw@nyu.edu>

if min(gtframelabs) == 0
  gtframelabs = gtframelabs + 1;
end
if min(estframelabs) == 0
  estframelabs = estframelabs + 1;
end

Pgt = 0;
Pest = 0;
Pboth = 0;
nfrm = length(estframelabs);
for n = 1:nfrm
  for m = 1:nfrm
    gtmatch = gtframelabs(n) == gtframelabs(m);
    estmatch = estframelabs(n) == estframelabs(m);

    if gtmatch;  Pgt = Pgt + 1;  end
    if estmatch;  Pest = Pest + 1;  end
    if gtmatch & estmatch;  Pboth = Pboth + 1;  end
  end
end

ppr = Pboth / Pest;
prr = Pboth / Pgt;
pfm = 2 * ppr * prr / (ppr + prr);
