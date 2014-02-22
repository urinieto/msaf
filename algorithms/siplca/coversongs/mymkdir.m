function r = mymkdir(dir)
% r = mymkdir(dir)
%   Ensure that dir exists by creating all its parents as needed.
% 2006-08-06 dpwe@ee.columbia.edu

[x,m,i] = fileattrib(dir);
if x == 0
  [pdir,nn,ee,vv] = fileparts(dir);
  mymkdir(pdir);
  disp(['creating ',dir,' ...']);
  mkdir(pdir, nn);
end
