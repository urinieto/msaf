function distmatrixwrite(matrix,rownames,colnames,filename,tag)
% distmatrixwrite(matrix,rownames,colnames,filename,tag)
%   Write a distance matrix
% 2006-08-06 dpwe@ee.columbia.edu

if nargin < 5
  tag = 'dpweDefault';
end

fid = fopen(filename, 'w');

fprintf(fid, '%s\n', tag);

usecolnames = zeros(length(colnames));

nn = 0;

for i = 1:length(rownames);
  nn = nn+1;
  fprintf(fid, '%d\t%s\n', nn, rownames{i});
  % if this name also occurs as a col, remember we don't need it
  usecolnames(strcmp(colnames, rownames(i))) = nn;
end

for i = 1:length(colnames);
  if usecolnames(i) == 0
    nn = nn+1;
    fprintf(fid, '%d\t%s\n', nn, colnames{i});
    usecolnames(i) = nn;
  end
end

% Matrix heading
fprintf(fid, 'Q/R');
for i = 1:length(usecolnames);
  fprintf(fid, '\t%d', usecolnames(i));
end
fprintf(fid, '\n');

% Rows of matrix
for i = 1:size(matrix,1);
  fprintf(fid, '%d', i);
  fprintf(fid, '\t%f', matrix(i,:));
  fprintf(fid, '\n');
end
