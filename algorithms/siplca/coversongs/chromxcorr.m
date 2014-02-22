function r = chromxcorr(A,F,L)
% r = chromxcorr(A,F,L)
%   Cross-correlate two chroma ftr vecs in both time and
%   transposition
%   Both A and F can be long, result is full convolution
%   (length(A) + length(F) - 1 columns, in F order).
%   L is the maximum lag to search to - default 100.
%   of shorter, 2 = by length of longer
%   Optimized version.
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

if nargin < 3;  L = 100; end

[nchr,nbts1] = size(A);
[nchr2,nbts2] = size(F);

if nchr ~= nchr2
  error('chroma sizes dont match');
end

r = zeros(nchr, 2*L+1);

for i = 1:nchr
  rr = 0;
  for j = 1:nchr
    rr = rr + xcorr(F(1+rem(j+i-2,nchr),:),A(j,:),L);
  end
  r(i,:) = rr;
end

% Normalize by shorter vector so max poss val is 1
r = r/min(nbts1,nbts2);
