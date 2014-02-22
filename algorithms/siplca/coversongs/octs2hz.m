function hz = octs2hz(octs,A440)
% hz = octs2hz(octs,A440)
% Convert a real-number octave 
% into a frequency in Hzfrequency in Hz into a real number counting 
% the octaves above A0. So hz2octs(440) = 4.0.
% Optional A440 specifies the Hz to be treated as middle A (default 440).
% 2006-06-29 dpwe@ee.columbia.edu for fft2chromamx

if nargin < 2;   A440 = 440; end

% A4 = A440 = 440 Hz, so A0 = 440/16 Hz

hz = (A440/16).*(2.^octs);


