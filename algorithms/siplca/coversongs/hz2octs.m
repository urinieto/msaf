function octs = hz2octs(freq, A440)
% octs = hz2octs(freq, A440)
% Convert a frequency in Hz into a real number counting 
% the octaves above A0. So hz2octs(440) = 4.0
% Optional A440 specifies the Hz to be treated as middle A (default 440).
% 2006-06-29 dpwe@ee.columbia.edu for fft2chromamx

if nargin < 2;   A440 = 440; end

% A4 = A440 = 440 Hz, so A0 = 440/16 Hz
octs = log(freq./(A440/16))./log(2);

