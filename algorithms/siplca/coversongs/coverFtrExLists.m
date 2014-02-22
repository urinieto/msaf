function [qfiles,tfiles] = coverFtrExLists(querylist, reflist, cachedir, skip, rootdir, ctype)
% [qfiles,tfiles] = coverFtrExLists(querylist, reflist, cachedir [, skip, rootdir, ctype])
%     Perform initial feature extraction for cover song contest.
%     querylist is a list of wav files that are the target of each test
%     reflist is a list of wav files that are the database to search
%     cachedir is the directory for storing temporary files.
%     skip is how many files were already done, so skip over them
%     this time.
%     optional rootdir is prepended to filenames read from
%     querylist, reflist.
%     ctype is passed down to chroma calculation for different
%     chroma types (default 1).
%     Specify reflist as '' if only one list.
%     After running, used coverDistMx.m to calculate the full
%     song-to-song distance matrix.
%     qfiles, tfiles are cell array lists of save data file names.
% 2006-08-06 dpwe@ee.columbia.edu for MIREX cover song contest

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

if nargin < 3; cachedir = 'cache_dir'; end
if nargin < 4; skip = 0; end
if nargin < 5; rootdir = ''; end
if nargin < 6; ctype = 1; end

tic;

qftrlistname = fullfile(cachedir, 'query.txt');
tftrlistname = fullfile(cachedir, 'test.txt');
rownamesfile = fullfile(cachedir, 'rownames.txt');
colnamesfile = fullfile(cachedir, 'colnames.txt');

% copy list files to cache dir
mymkdir(cachedir);
listfilewrite(listfileread(querylist),rownamesfile);
listfilewrite(listfileread(reflist),colnamesfile);


% Do feature extraction on the query songs

% ftrex params
div = 1;
%fminl = 0;
%fminu = 300;
%fmaxl = 500;
%fmaxu = 1000;

fctr = 400;
fsd = 1.0;

if skip > 0
  % We already processed some files, read those names in
  qfiles = listfileread(qftrlistname);
  qfiles = qfiles(1:min(skip,length(qfiles)));
else
  % we're starting from scratch
  qfiles = [];
  % make sure there are no left-over list files
  if fexist(qftrlistname)
    delete(qftrlistname);
  end
  if fexist(tftrlistname)
    delete(tftrlistname);
  end
end

if length(qfiles) < skip
  % Already processed into the test files...?
  tfiles = listfileread(tftrlistname);
  tfiles = tfiles(1:(skip - length(qfiles)));
else
  tfiles = [];
end

if length(qfiles) == skip
  qfiles = [qfiles, calclistftrs(querylist,rootdir,'',cachedir,'.mat', ...
	                        skip, fctr, fsd, ctype)];
end

if length(reflist) > 0
  tfiles = [tfiles, calclistftrs(reflist,rootdir,'',cachedir,'.mat', ...
	                        max(0,skip-length(qfiles)),...
                                fctr, fsd, ctype)];
end

listfilewrite(qfiles, qftrlistname);

% Only write tfiles if there are any
if length(tfiles) > 0
  listfilewrite(tfiles, tftrlistname);
end

toc
