function opfiles = calclistftrs(listfile,srcprepend,srcext,dstprepend,dstext,skip,fctr, fsd, ctype)
% calclistftrs(listfile,srcprepend,srcext,dstprepend,dstext,fctr,fsd,ctype)
%   Take listile, a list of input MP3 or WAV files and calculate 
%   beat-synchronous chroma features for each one. 
%   input file names each have srcprepend prepended and srcext appended;
%   features are written to .mat files with the same root name but
%   with dstprepend prepended and dstext appended.  First <skip>
%   items are skipped (for resumption of interrupted runs).
%   fctr and fsd specify a spectral window used to extract chroma 
%   elements with center at fctr Hz, and gaussian log-F half-width
%   of fsd octaves.
%   Return a cell array of the output files written.
% 2006-07-14 dpwe@ee.columbia.edu

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

if nargin < 2; srcprepend = ''; end
if nargin < 3; srcext = ''; end
if nargin < 4; dstprepend = ''; end
if nargin < 5; dstext = ''; end

if nargin < 6; skip = 0; end
% downweight fundamentals below here
if nargin < 7; fctr = 400; end
if nargin < 8; fsd = 1.0; end
if nargin < 9; ctype = 1; end

[files,nfiles] = listfileread(listfile);

if nfiles < 1
  error(['No sound file names read from list file "',listfile,'"']);
end

for songn = 1:nfiles
  tline = files{songn};

  if length(srcext) > 0
    if tline(end-length(srcext)+1:end) == srcext
      % chop off srcext already there
      tline = tline(1:end-length(srcext));
    end
  else
    % no srcext specified - must be part of input file name
    % separate name and extension for input file
    [srcpath, srcname, srcext, srcvsn] = fileparts(tline);
    tline = fullfile(srcpath,srcname);
  end
    
  % lines are 
  ifname = fullfile(srcprepend,[tline,srcext]);
  ofname = fullfile(dstprepend,[tline,dstext]);
  
  if songn > skip
  
    disp(['song ',num2str(songn),' ',ifname,' -> ',ofname]);

    [ofdir,nn,ee,vv] = fileparts(ofname);
    % Make sure the parent directory exists
    mymkdir(ofdir)
    
    % wav files or mp3 files
    if srcext == '.mp3'
      [d,sr]=mp3read(ifname,0,1,2); 
    else
      [d,sr]=wavread(ifname);
    end
    [F,bts] = chrombeatftrs(d,sr,fctr,fsd,ctype);
    save(ofname,'ifname','F','bts');
    disp([datestr(rem(now,1),'HH:MM:SS'), ' ', ifname,' ncols=',num2str(size(F,2)),' bpm=',num2str(60/median(diff(bts)))]);

  end
  
  opfiles{songn} = ofname;
  
end
