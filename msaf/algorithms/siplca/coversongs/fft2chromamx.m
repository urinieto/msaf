function wts = fft2chromamx(nfft,nbins,sr,A440,ctroct,octwidth)
% wts = fft2chromamx(nfft,nbins,sr,A440,ctroct,octwidth)
%     Create a wts matrix to convert FFT to Chroma
%     A440 is optional ref frq for A
%     ctroct, octwidth specify a dominance window - Gaussian
%     weighting centered on ctroct (in octs, re A0 = 27.5Hz) and 
%     with a gaussian half-width of octwidth.  Defaults to
%     halfwidth = inf i.e. flat.
%     2006-06-29 dpwe@ee.columbia.edu

if nargin < 2;   nbins = 12;  end
if nargin < 3;   sr = 22050; end
if nargin < 4;   A440 = 440; end
if nargin < 5;   ctroct = 5; end
if nargin < 6;   octwidth = 0; end

wts = zeros(nbins, nfft);

fftfrqbins = nbins*hz2octs([1:(nfft-1)]/nfft*sr,A440);

% make up a value for the 0 Hz bin = 1.5 octaves below bin 1
% (so chroma is 50% rotated from bin 1, and bin width is broad)
fftfrqbins = [fftfrqbins(1)-1.5*nbins,fftfrqbins];

binwidthbins = [max(1, fftfrqbins(2:nfft) - fftfrqbins(1:(nfft-1))), 1];

D = repmat(fftfrqbins,nbins,1) - repmat([0:(nbins-1)]',1,nfft);

nbins2 = round(nbins/2);

% Project into range -nbins/2 .. nbins/2
% add on fixed offset of 10*nbins to ensure all values passed to rem are +ve
D = rem(D + nbins2 + 10*nbins, nbins) - nbins2;

% Gaussian bumps - 2*D to make them narrower
wts = exp(-0.5*(2*D./repmat(binwidthbins,nbins,1)).^2);

% normalize each column
wts = wts./repmat(sqrt(sum(wts.^2)),nbins,1);

% remove aliasing columns
wts(:,[(nfft/2+2):nfft]) = 0;

% Maybe apply scaling for fft bins
if octwidth > 0
  wts = wts.*repmat(exp(-0.5*(((fftfrqbins/nbins - ctroct)/octwidth).^2)), nbins, 1);
end

%wts = binwidthbins;
%wts = fftfrqbins;

function octs = hz2octs(freq, A440)
% octs = hz2octs(freq, A440)
% Convert a frequency in Hz into a real number counting 
% the octaves above A0. So hz2octs(440) = 4.0
% Optional A440 specifies the Hz to be treated as middle A (default 440).
% 2006-06-29 dpwe@ee.columbia.edu for fft2chromamx

%if nargin < 2;   A440 = 440; end

% A4 = A440 = 440 Hz, so A0 = 440/16 Hz
octs = log(freq./(A440/16))./log(2);


