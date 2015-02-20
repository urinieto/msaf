function E = fexist(F)
%  E = fexist(F)  returns 1 if file F exists, else 0
% 2006-08-06 dpwe@ee.columbia.edu

x = dir(F);

E = length(x);
