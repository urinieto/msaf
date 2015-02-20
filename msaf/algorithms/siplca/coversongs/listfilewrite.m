function N = listfilewrite(L, F)
% N = listfilewrite(L, F)   Write a list of items to a file
%    L is a cell array of strings.  Write to file F, one per line.
%    N returns the number of items successfully written.
% 2006-08-06 dpwe@ee.columbia.edu  for MIREX 06

fid = fopen(F, 'w');

nit = length(L);

for i = 1:nit
  fprintf(fid, '%s\n', L{i});
end

fclose(fid);

