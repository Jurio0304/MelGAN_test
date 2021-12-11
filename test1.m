x = audioread('D:\Python\paper_12\test_result\original_2.wav');
y = audioread('D:\Python\paper_12\test_result\generated_2.wav');
z = audioread('D:\Python\paper_12\reco_generate_result\original_2.wav');
w = audioread('D:\Python\paper_12\reco_generate_result\reco_generated_2.wav');

subplot(4,1,1)
plot(x);
title('wav_-2')
xlabel('original')
hold on
subplot(4,1,2)
plot(y);
xlabel('paper12_-generated')
hold on
subplot(4,1,3)
plot(z);
xlabel('paper7_-reco')
hold on
subplot(4,1,4)
plot(w);
xlabel('reco->generated')
hold off