%-- Unknown date --%
tic; vp5 = (viterbi_path(prtx(1,:), prtx, obsprb))' -1; toc
[mean(gt == vp5) mean(gt(gtnz) == vp5(gtnz)) mean((gt>0) == (vp5>0)) mean(vp5>0)]
obsprb(1,:) = .45 - .2*pdp';
tic; vp5 = (viterbi_path(prtx(1,:), prtx, obsprb))' -1; toc
[mean(gt == vp5) mean(gt(gtnz) == vp5(gtnz)) mean((gt>0) == (vp5>0)) mean(vp5>0)]
pr1n = (pr1 - mean(pr1))/std(pr1);
mean(pr1n)
std(pr1n)
obsprb(1,:) = .45 - .2*pdp' + .1 * pr1n;
tic; vp5 = (viterbi_path(prtx(1,:), prtx, obsprb))' -1; toc
[mean(gt == vp5) mean(gt(gtnz) == vp5(gtnz)) mean((gt>0) == (vp5>0)) mean(vp5>0)]
obsprb(1,:) = .45 - .2*pdp' + .01 * pr1n;
tic; vp5 = (viterbi_path(prtx(1,:), prtx, obsprb))' -1; toc
[mean(gt == vp5) mean(gt(gtnz) == vp5(gtnz)) mean((gt>0) == (vp5>0)) mean(vp5>0)]
obsprb(1,:) = .45 - .2*pdp' - .01 * pr1n;
tic; vp5 = (viterbi_path(prtx(1,:), prtx, obsprb))' -1; toc
[mean(gt == vp5) mean(gt(gtnz) == vp5(gtnz)) mean((gt>0) == (vp5>0)) mean(vp5>0)]
obsprb(1,:) = .45 - .2*pdp' - .0 * pr1n;
tic; vp5 = (viterbi_path(prtx(1,:), prtx, obsprb))' -1; toc
[mean(gt == vp5) mean(gt(gtnz) == vp5(gtnz)) mean((gt>0) == (vp5>0)) mean(vp5>0)]
for b = .4:.01:.5;  obsprb(1,:) = b - .2*pdp'; vp5 = (viterbi_path(prtx(1,:), prtx, obsprb))' -1; [b mean(gt == vp5) mean(gt(gtnz) == vp5(gtnz)) mean((gt>0) == (vp5>0)) mean(vp5>0)]; end
disp(num2str(rand(1,3)))
for b = .4:.01:.5;  obsprb(1,:) = b - .2*pdp'; vp5 = (viterbi_path(prtx(1,:), prtx, obsprb))' -1; disp(num2str([b mean(gt == vp5) mean(gt(gtnz) == vp5(gtnz)) mean((gt>0) == (vp5>0)) mean(vp5>0)])); end
for c=.15:.01:.25;  obsprb(1,:) = .47 - c*pdp'; vp5 = (viterbi_path(prtx(1,:), prtx, obsprb))' -1; disp(num2str([c mean(gt == vp5) mean(gt(gtnz) == vp5(gtnz)) mean((gt>0) == (vp5>0)) mean(vp5>0)])); end
for c=.25:.01:.35;  obsprb(1,:) = .47 - c*pdp'; vp5 = (viterbi_path(prtx(1,:), prtx, obsprb))' -1; disp(num2str([c mean(gt == vp5) mean(gt(gtnz) == vp5(gtnz)) mean((gt>0) == (vp5>0)) mean(vp5>0)])); end
for c=.05:.01:.2;  obsprb(1,:) = c - .9*c*pdp'; vp5 = (viterbi_path(prtx(1,:), prtx, obsprb))' -1; disp(num2str([c mean(gt == vp5) mean(gt(gtnz) == vp5(gtnz)) mean((gt>0) == (vp5>0)) mean(vp5>0)])); end
for c=.01:.01:.1;  obsprb(1,:) = c - .9*c*pdp'; vp5 = (viterbi_path(prtx(1,:), prtx, obsprb))' -1; disp(num2str([c mean(gt == vp5) mean(gt(gtnz) == vp5(gtnz)) mean((gt>0) == (vp5>0)) mean(vp5>0)])); end
for c=.00:.01:.1;  obsprb(1,:) = c - .9*c*pdp'; vp5 = (viterbi_path(prtx(1,:), prtx, obsprb))' -1; disp(num2str([c mean(gt == vp5) mean(gt(gtnz) == vp5(gtnz)) mean((gt>0) == (vp5>0)) mean(vp5>0)])); end
%--  5:23 PM 8/01/05 --%
tic; trysmooth; toc
%--  2:13 PM 8/08/05 --%
expts
whos
save
results(1)
results(5)
1216/60
%-- 10:49 AM 8/14/05 --%
expts
save
whos
results(1,1)
results(1,10)
results(2,10)
results(3,10)
results(4,10)
results(4,1)
results(4,2)
results(4,3)
results(4,4)
results(4,5)
results(4,6)
results(4,7)
results(4,8)
results(4,9)
whos
%--  4:57 PM 8/15/05 --%
load
whos
clear results
expts
whos
%-- 9/20/05  4:26 PM --%
%--  3:52 PM 9/22/05 --%
makeTrnData
!mv ../makeTrnData.m .
makeTrnData
makeTestData
ls
!ls
!rm ../makeTestData.m
makeTestData
makeTrnData
makeTestData
~ls
!ls
cd hmm2
ls
clear ls
!ls
load mirexRBF.mat
mirexRBF
makeTestData
pwd
cd ..
makeTestData
fg
makeTestData
!ls
ls trainData
!ls trainData
makeTrnData
makeTestData
whos
%-- 12:22 PM 8/28/06 --%
[d,sr] = mp3read('LIB.mp3',0,1,4);
d = resample(d,8,11);
sr = 8000;
b = beattrack(d,sr);
b = beattrackdp(d,sr);
tic; b = beattrackdp(d,sr); toc
size(b)
sum(b)
tic; b = beattrack(d,sr); toc
%-- 10:38 PM 9/09/06 --%
clear
load 01-Know_Your_Rights-chrft.mat
whos
b
bts
median(diff(bts))
b
ls
whos
ifname
clear
load 01-Know_Your_Rights-chrft.mat
whos
cd ../..
pwd
D = load('clash/Combat_Rock/01-Know_Your_Rights.mp3');
pwd
D = load('clash/Combat_Rock/01-Know_Your_Rights.mat');
ls clash
ls clash/Combat_Rock
pwd
D = load('clash/Combat_Rock/01-Know_Your_Rights-chrft.mat');
D
%-- 9/27/06  4:58 PM --%
ls
dpweCoverFtrEx2Lists('mp3listQ.txt','mp3listT.txt','cache',0,'/homes/dpwe/projects/coversongs/covers');
dpweCoverDistMx('cache','distMx.txt');
dm = dpweCoverDistMx('cache');
imgsc(dm)
gcolor
[vv,xx] = min(dm');
xx
sum(xx==1:15)
dm0 = dm
dpweCoverFtrEx2Lists('mp3listQ.txt','mp3listT.txt','cache2',0,'/homes/dpwe/projects/coversongs/covers');
dm = dpweCoverDistMx('cache2');
imgsc(dm)
imgsc(dm0)
imgsc(dm)
imgsc(dm0)
imgsc(dm)
[vv,xx] = min(dm');
sum(xx==1:15)
subplot(221)
imgsc(dm0)
subplot(222)
imgsc(dm)
caxis
caxis([0 150])
subplot(221)
caxis([0 150])
whos
load('cache/Abracadabra/steve_miller_band+Steve_Miller_Band+09-Abracadabra.mat');
pwd
load('cache/Abracadabra/steve_miller_band+Steve_Miller_Band_Live_+09-Abracadabra.mat');
whos
subplot(411)
imgsc(F)
subplot(412)
load('cache2/Abracadabra/steve_miller_band+Steve_Miller_Band_Live_+09-Abracadabra.mat');
imgsc(F)
subplot(223)
imgsc(dm0)
subplot(224)
imgsc(dm)
load('cache/Faith/george_michael+Faith+01-Faith.mat');
subplot(411)
imgsc(F)
subplot(412)
load('cache2/Faith/george_michael+Faith+01-Faith.mat');
imgsc(F)
size(b)
b
size(bts)
bts
load('cache/Faith/george_michael+Faith+01-Faith.mat');
b
size(b)
[d,sr] = mp3read('covers/Faith/george_michael+Faith+01-Faith.mat',0,1,2);
[d,sr] = mp3read('covers/Faith/george_michael+Faith+01-Faith.mp3',0,1,2);
bts = beat(d,sr,[160 190],1,6,0.8,0,240,1.5)
bts = beat(d,sr,0,1,6,0.8,0,240,1.5)
bts = beat(d,sr,[160 190],1,6,0.8,[],0,240,1.5)
bts = beattrack(d,sr,[160 190],1);
bts = beat(d,sr,[160 190],1,6,0.8,[],0,240,1.5);
bts = beat(d,sr,[0 20],1,6,0.8,[],0,240,1.5);
bts = beattrack(d,sr,[0 20],1);
bts = beat(d,sr,[0 20],1,6,0.8,[],0,240,1.5);
subplot(312)
bts = beat(d,sr,[0 20],1,6,0.8,[],0,240,1.5);
subplot(412)
axis([0 20 -1 4])
subplot(413)
axis([0 20 -1 4])
bts = beat(d,sr,[20 40],1,6,0.8,[],0,240,1.5);
dpweCoverFtrEx2Lists('mp3listQ.txt','mp3listT.txt','cache2',0,'/homes/dpwe/projects/coversongs/covers');
dm = dpweCoverDistMx('cache2');
[vv,xx] = min(dm');
sum(xx==1:15)
subplot(221)
imgsc(dm0)
colorbar
subplot(222)
imgsc(dm)
colorbar
caxis([0 90])
[vv0,xx0] = min(dm0');
xx==1:15
xx0==1:15
dpweCoverFtrEx2Lists2T('mp3listQ.txt','mp3listT.txt','cache2T',0,'/homes/dpwe/projects/coversongs/covers');
dm2T = dpweCoverDistMx('cache2T');
subplot(223)
imgsc(dm2T)
subplot(224)
imgsc(min22(dm2T))
[vv2T,xx2T] = min(min22(dm2T)');
sum(xx2T==1:15)
(xx2T==1:15)
(xx==1:15)
dpweCoverFtrEx2Lists('mp3listQ.txt','mp3listT.txt','cache2E',0,'/homes/dpwe/projects/coversongs/covers',2);
dm2E = dpweCoverDistMx('cache2E');
[vv2E,xx2E] = min(min22(dm2E)');
sum(xx2E==1:15)
size(dm2E)
[vv2E,xx2E] = min((dm2E)');
sum(xx2E==1:15)
xx2E==1:15
subplot(411)
load('cache/Faith/george_michael+Faith+01-Faith.mat');
imgsc(F)
subplot(412)
load('cache2E/Faith/george_michael+Faith+01-Faith.mat');
imgsc(F)
[B,A] = butter(3,.1);
plot(filter(B,A,[1,zeros(1,100)]))
subplot(111)
plot(filter(B,A,[1,zeros(1,100)]))
plot(grpdelay(B,A,1024,20000))
freqz(B,A)
plot(filter(B,A,[1,zeros(1,100)]))
cd ~/tmp/suman
ls
[d1,sr] = wavread('f1nw_1.wav');
[d0,sr] = wavread('f1nw_1_orig.wav');
subplot(211);specgram(d0,512,sr)
subplot(212);specgram(d1,512,sr)
axis([0 3.2 0 1000])
subplot(211);
axis([0 3.2 0 1000])
help yin
yin(d0,sr)
figure
yin(d1,sr)
axis([0 3.2 -1.5 -.5)
axis([0 3.2 -1.5 -.5])
pwd
cd ~/projects/coversongs/mirex06results
ls
R = textread('countsnotext.txt');
size(R)
sum(R)
Rde = R(:,4);
Rde = reshape(Rde, 10,33);
hist(sum(Rde))
size(Rde)
size(sum(Rde)
size(sum(Rde))
sum(Rde)
subplot(212)
Rxx = R(:,5);
sum(Rxx)
Rxx = reshape(Rxx, 10,33);
hist(sum(Rxx))
hist(sum(Rxx),0:10:70)
hist(sum(Rxx),5:10:70)
hist(sum(Rxx),5:5:70)
hist(sum(Rxx),1:5:70)
hist(sum(Rxx),2.5:5:70)
subplot(211)
hist(sum(Rde),2.5:5:70)
subplot(212)
hist(Rde(:))
subplot(211)
hist(Rde(:))
subplot(212)
hist(Rxx(:))
imgsc(R(:,3:10))
subplot(211)
subplot(121)
imgsc(R(:,3:10))
gcolor
colorbar
set(gca,'XTick',[1:8])
set(gca,'XTickLabel','CS|DE|KL1|KL2|KWL|KWT|LR|TP')
set(gca,'YTick',[10:10:320])
set(gca,'YTick',[0:10:320])
set(gca,'YTick',[1:10:320])
set(gca,'YTick',[10:10:320])
set(gca,'YTickLabel',[1:1:32])
title('Raw Retrieval per song')
grid
print -adobecset -depsc raw-retrieval.eps
persong = max(R(:,3:10));
persong = max(R(:,3:10)');
perssong = sum(reshape(persong,10,33));
[vv,xx] = sort(-perssong);
xx
xxx = repmat(10*(xx - 1),1,10) + repmat([1:10]',33,1);
xxx = repmat(10*(xx - 1),10,1) + repmat([1:10]',1,33);
imgsc(R(xxx(:),3:10))
ls
cd ~/tmp
[d,sr = wavread('100_0016.wav');
[d,sr] = wavread('100_0016.wav');
length(d)/sr
subplot(211)
specgram(d,512,sr)
sr
subplot(212)
plot(d)
plot(d(1:2000))
plot(d(1:10000))
plot(d(1:20000))
plot(d(6001:10000))
subplot(211)
caxis
caxis([-40 40])
axis([0 38 0 1000])
specgram(d,2048,sr)
caxis([-40 40])
axis([0 38 0 1000])
pwd
cd ../projects/coversongs/dist
ls
[d,sr] = wavread('../mirex06data/train02.wav');
[d,sr] = wavread('../mirex06train/train02.wav');
ls ../mirex06train
[d,sr] = wavread('../mirex06train/train2.wav');
help tempo
[t,xcr,oe,sr] = tempo(d,sr);
t
b = beat(d,sr);
sr
[t,xcr,oe,sr] = tempo(d,sr);
[d,sr] = wavread('../mirex06train/train2.wav');
[t,xcr,D,oe,sr] = tempo(d,sr);
sr
t
[d,sr] = wavread('../mirex06train/train2.wav');
[t,xcr,D,oe,sgsr] = tempo(d,sr);
help tempo
help beat
[b,oeb,Db,cs] = beat(d,sr);
subplot(411)
imgsc(Db)
subplot(412)
plot(oeb)
help mkblips
db = mkblips(b,sr,length(d));
wavwrite((d+db)/2,sr,'/homes/dpwe/public_html/tmp/tmp/wav');
wavwrite((d+db)/2,sr,'/homes/dpwe/public_html/tmp/tmp.wav');
max(d)
max(db)
db = mkblips(b,sr,length(d));
wavwrite((d+db)/2,sr,'/homes/dpwe/public_html/tmp/tmp.wav');
gtb = textread('../mirex06train/train2.txt');
size(gtb)
gtb(1:10,1:10)
for i = gtb(:); bin = round(i/.010); if bin > 0; vv(bin) = vv(bin)+1; end
end
size(vv)
sum(vv)
max(gtb(:))
max(gtb(:))/.01
vv = zeroz(1,2989);
vv = zeros(1,2989);
for i = gtb(:); bin = round(i/.010); if bin > 0; vv(bin) = vv(bin)+1; end; end
max(vv)
min(vv)
help foreach
for i = 1:10; disp(num2str(i)); emd
end;
for i = 1:10; disp(num2str(i)); end
size(gtb(:))
sum(vv)
size(vv)
for i = gtb(:); bin = round(i/.010); if bin > 0; vv(bin) = vv(bin)+1; end; end
max(vv)
bin
for i = gtb(:)'; bin = round(i/.010); if bin > 0; vv(bin) = vv(bin)+1; end; end
max(vv)
plot(vv)
plot(vv(1:100))
help beat
b = beat(d,sr,0,0,[0 15]);
plot([0:0.01:15],vv(1:1501))
cvv = conv(vv,hann(5)'/sum(hann(5)));
cvv = cvv(2+[1:length(vv)]);
plot([0:0.01:15],cvv(1:1501))
cvv = conv(vv,hann(9)'/sum(hann(9)));
cvv = cvv(4+[1:length(vv)]);
plot([0:0.01:15],cvv(1:1501))
hold on; plot([b;b],[0;5],'-g')
axis([0 15 0 5])
b = beat(d,sr,0,0,[0 15]);
hold off
b = beat(d,sr,0,0,[0 15]);
axis([0 15 0 15])
axis([0 15 0 12])
axis([0 15 -2 12])
subplot(414)
plot([0:0.01:15],cvv(1:1501))
hold on; plot([b;b],[0;5],'-g'); hold off
axis([0 15 -1 5])
plot([b;b],[0;5],'-g');
hold on; plot([0:0.01:15],cvv(1:1501)); hold off
axis([0 15 -1 5])
plot([b;b],[-1;5],'-g');
hold on; plot([0:0.01:15],cvv(1:1501)); hold off
axis([0 15 -1 5])
sum(hann(9))
print -adobecset -depsc bragg-beats+gt.eps
subplot(411)
imgsc([0 15],[0 40],D)
imgsc([0 30],[0 40],D)
subplot(412)
plot(D(10,:))
plot(diff(D(10,:)))
plot(max(0,diff(D(10,:))))
subplot(411)
axis([0 5 0 40])
sgsr
plot([2:(5*250)]/sgsr,max(0,diff(D(10,1:(5*250)))))
subplot(411)
imgsc([0 30],[0 40],D)
subplot(412)
plot([2:(5*250)]/sgsr,max(0,diff(D(10,1:(5*250)))))
subplot(411)
axis([0 5 0 40])
subplot(412)
plot([2:(5*250)]/sgsr,max(0,diff(D(1,1:(5*250)))))
for i = 2:9; hold on; plot([2:(5*250)]/sgsr,i+0.1*max(0,diff(D(1,1:(5*250))))); hold off; end
plot([2:(5*250)]/sgsr,0.1*max(0,diff(D(1,1:(5*250)))))
for i = 1:9; hold on; plot([2:(5*250)]/sgsr,i+0.1*max(0,diff(D(1+4*i,1:(5*250))))); hold off; end
subplot(413)
plot([2:(5*250)]/sgsr,oe(1:(5*250)))
plot([1:(5*250)]/sgsr,oe(1:(5*250)))
subplot(414)
axis([0 5 -1 5])
subplot(413)
plot([1:(5*250)]/sgsr,oe(1000+[1:(5*250)]))
subplot(411)
axis([4 9 0 40])
subplot(414)
axis([4 9 -1 5])
subplot(412)
plot((1000+[2:(5*250)])/sgsr,0.1*max(0,diff(D(1,1000+[1:(5*250)]))))
for i = 1:9; hold on; plot((1000+[2:(5*250)])/sgsr,i+0.1*max(0,diff(D(1+4*i,1000+[1:(5*250)])))); hold off; end
grid
subplot(413)
plot((1000+[2:(5*250)])/sgsr,oe(1,1000+[1:(5*250)]))
plot((1000+[1:(5*250)])/sgsr,oe(1,1000+[1:(5*250)]))
grid
axis([4 9 -1 2.5])
print -adobecset -depsc bragg-onsetenv-4-9.eps
[t,xcr,D,oe,sgsr] = tempo(d,sr);
[t,xcr,D,oe,sgsr] = tempo(d,sr,120,1,[],1);
subplot(414)
axis([0 500 -200 200])
set(gca,'XTickLabel',[0:50:500]*.004)
60/168.53
print -adobecset -depsc bragg-onsetenv-4-9+autoco.eps
subplot(411)
axis([10 15 0 40])
subplot(413)
plot((2500+[1:(5*250)])/sgsr,oe(1,2500+[1:(5*250)]))
axis([10 15 -1 3])
subplot(411)
axis([10 14 0 40])
subplot(413)
plot((2500+[1:(4*250)])/sgsr,oe(1,2500+[1:(4*250)]))
axis([10 14 -1 3])
subplot(412)
plot((2500+[2:(4*250)])/sgsr,0.1*max(0,diff(D(1,2500+[1:(4*250)]))))
for i = 1:9; hold on; plot((2500+[2:(4*250)])/sgsr,i+0.1*max(0,diff(D(1+4*i,2500+[1:(4*250)])))); hold off; end
grid
subplot(413)
grid
subplot(414)
axis([0 1000 -200 200])
set(gca,'XTick',[0:200:1600])
set(gca,'XTick',[0:125:1000])
set(gca,'XTickLabel',[0:125:1000]*0.004)
print -adobecset -depsc bragg-onsetenv-10-14+autoco.eps
plot([0:0.01:15],cvv(1:1501))
