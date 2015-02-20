function mx = coverDistMxLists(cachedir, opdistfile)
% mx = coverDistMxLists(...)
%    Calculate the cover song dist matrix.  
%    mx = coverDistMxLists(cachedir)  reads all details from cachedir, as 
%      written by coverFtrExLists
%    coverDistMxLists(..., opdistfile)  writes the distance matrix to 
%      the named file in standard MIREX distance matrix format.
%    mx returns the distance matrix, one row per query song, one
%    column per test song, query and test as passed to coverFtrEx.m.
% 2006-08-06 dpwe@ee.columbia.edu  for mirex 06

%   Copyright (c) 2006 Columbia University.
% 
%   This file is part of LabROSA-coversongID
% 
%   LabROSA-coversongID is free software; you can redistribute it and/or modify
%   it under the terms of the GNU General Public License version 2 as
%   published by the Free Software Foundation.
% 
%   LabROSA-coversongID is distributed in the hope that it will be useful, but
%   WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%   General Public License for more details.
% 
%   You should have received a copy of the GNU General Public License
%   along with LabROSA-coversongID; if not, write to the Free Software
%   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
%   02110-1301 USA
% 
%   See the file "COPYING" for the text of the license.

havetlist = 0;
if nargin < 1;  cachedir = 'cache_dir';  end
if nargin < 2;  opdistfile = ''; end

% Where coverFtrEx saved the details of the queries/tests
qftrlistname = fullfile(cachedir, 'query.txt');
tftrlistname = fullfile(cachedir, 'test.txt');
rownamesfile = fullfile(cachedir, 'rownames.txt');
colnamesfile = fullfile(cachedir, 'colnames.txt');

if exist(tftrlistname) == 0
  % no separate test file - use qfile again
  tftrlistname = qftrlistname;
else
  havetlist = 1;
end

% params
pwr = 0.5;
metric = 2;
verb = 1;

[R,S,T] = coverTestLists(readlistfile(qftrlistname), ...
                         readlistfile(tftrlistname), ...
			 pwr, metric, verb);

% write dist matrix
if length(opdistfile) > 0
  rownames = listfileread(rownamesfile);
  if havetlist
    colnames = listfileread(colnamesfile);
  else
    colnames = rownames;
  end
  distmatrixwrite(1./R,rownames,colnames,opdistfile,'dpweCover20060806');
end

if nargout > 0
  mx = 1./R;
end

