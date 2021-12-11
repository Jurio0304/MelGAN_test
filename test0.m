x = audioread('D:\Python\paper_12\test_result\original_3.wav');
y = audioread('D:\Python\paper_12\test_result\generated_3.wav');

subplot(2,1,1)
plot(x);
title('wav_-3')
xlabel('original')
hold on
subplot(2,1,2)
plot(y);
xlabel('generated')

hold off