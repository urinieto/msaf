function [L,N] = listfileread(F)
% [L,N] = listfileread(F)   Read a list of per-line items
%    F is a file containing a list of items, one per line.
%    Return L as a cell array, with one item per line, N as the
%    number of items.
%    If F is not found, return empty L and N == -1 (instead of 0).
% 2006-08-06 dpwe@ee.columbia.edu for MIREX 06

N = -1;
L = [];

if fexist(F) == 1

  fid = fopen(F);

  nitems = 0;

  while 1
    tline = fgetl(fid);
    if ~ischar(tline), break, end

    nitems = nitems+1;
    L{nitems} = tline;
  end
  fclose(fid);
  N = nitems;

end
