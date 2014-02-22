function Y = chromagram_P(d,sr,fftlen,nbin,f_ctr,f_sd)
% Y = chromagram_E(d,sr,fftlen,nbin)
%  Calculate a "chromagram" of the sound in d (at sampling rate sr)
%  Use windows of fftlen points, hopped by ffthop points
%  Divide the octave into nbin steps
%  Weight with center frequency f_ctr (in Hz) and gaussian SD f_sd (in octaves)
% 2006-09-26 dpwe@ee.columbia.edu

if nargin < 3;   fftlen = 2048; end
if nargin < 4;   nbin = 12; end
if nargin < 5;   f_ctr = 1000; end
if nargin < 6;   f_sd = 1; end

fftwin = fftlen/2;
ffthop = fftlen/4;  % always for this

D = abs(specgram(d,fftlen,sr,fftwin,(fftwin-ffthop)));

[nr,nc] = size(D);

A0 = 27.5; % Hz
A440 = 440; % Hz

f_ctr_log = log(f_ctr/A0) / log(2);

CM = fft2chromamx(fftlen, nbin, sr, A440, f_ctr_log, f_sd);
% Chop extra dims
CM = CM(:,1:(fftlen/2)+1);

% Keep only local maxes in freq
Dm = (D > D([1,[1:nr-1]],:)) & (D >= D([[2:nr],nr],:));
Y = CM*(D.*Dm);
