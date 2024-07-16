% Input:
% H_full [64 x 160 x 50,000]

clear; clc;
rng(47);

% Parameters
na = 64;                % # of BS antennas
nc = 160;               % # of OFDM subcarriers
nTrain = 40000;          % # of training samples
nVal = 5000;
nTest = 5000;           % # of test samples
snrTrain = 10;          % Noise level in training samples. Value in linear units: -1=infdB or 1=0dB, 10=10dB, 1000=30dB
snrTest = 10;           % Noise level in test samples. Value in linear units: -1=infdB or 1=0dB, 10=10dB, 1000=30dB

%% Import and preprocess data
fprintf('Importing and preprocessing data...\n')

H_full_real = load('data/combined_data.mat').real_part; 
H_full_imag = load('data/combined_data.mat').imag_part; 
H_full = complex(H_full_real, H_full_imag); % complex 64x160x60000
H_full_check = H_full(:, :, 1);
% Split data
H_train = H_full(:, :, 1:nTrain);
H_val = H_full(:, :, nTrain+1:nTrain+nVal);
H_test = H_full(:, :, nTrain+nVal+1:nTrain+nVal+nTest);

% UL Training
HUL_train_n = H_train;
Lambda = squeeze(1 ./ mean(abs(HUL_train_n).^2, [1 2])); % tính trung bình bình phương giá trị tuyệt đối của ma trận kênh HUL_train_n hoặc HDL_test_n trên các chiều không gian (chiều 1 và 2), sau đó nghịch đảo của giá trị trung bình này được tính.
if snrTrain ~= -1 % MUC DICH: xử lý một tập dữ liệu (H_train) bằng cách thêm nhiễu Gaussian trắng (HN) và cập nhật giá trị của Lambda dựa trên dữ liệu đã xử lý (HUL_train_n)
    nPower = 1 ./ (Lambda * snrTrain); % tinh cong suat nhieu
    HN = bsxfun(@times, randn(na, nc, nTrain) + 1i * randn(na, nc, nTrain), reshape(sqrt(nPower / 2), 1, 1, [])); % nhieu Gauss (HN) duoc tao nen bang cach nhan một ma trận ngẫu nhiên phức voi ma trận (1,1,nTrain) chứa giá trị sqrt(nPower/2)
    HUL_train_n = H_train + HN; % ma tran kenh co nhieu
    Lambda = squeeze(1 ./ mean(abs(HUL_train_n).^2, [1 2])); % Tinh lai Lambda
end
HUL_train_n = bsxfun(@times, HUL_train_n, reshape(sqrt(Lambda), 1, 1, [])); % nhân ma trận HUL_train_n với ma trận sqrt(Lambda) đã được định dạng lại thành kích thước (1,1,nTrain)
HUL_train_compl_tmp = reshape(HUL_train_n, na * nc, nTrain).'; % thay đổi hình dạng của ma trận HUL_train_n từ kích thước (na, nc, nTrain) thành ma trận 2D có kích thước (nTrain, na*nc)
HUL_train_compl_tmp_mean = mean(HUL_train_compl_tmp); % tính trung bình
HUL_train_compl = bsxfun(@minus, HUL_train_compl_tmp, HUL_train_compl_tmp_mean); % trừ phần tử giữa ma trận HUL_train_compl_tmp với giá trị trung bình

% DL Validation
HDL_val_n = H_val;
Lambda = squeeze(1 ./ mean(abs(HDL_val_n).^2, [1 2])); % tính nghịch đảo của giá trị trung bình bình phương phần tuyệt đối của HDL_val_n qua các chiều 1 và 2
HDL_val = bsxfun(@times, HDL_val_n, reshape(sqrt(Lambda), 1, 1, [])); % nhân từng phần tử của HDL_val_n với mảng sqrt(Lambda) đã được định hình lại thành kích thước (1,1,nVal)
if snrTest ~= -1 % MUC DICH: thêm nhiễu Gaussian vào ma trận H_val để tạo ra HDL_val_n với công suất nhiễu dựa trên snrTest
    for q = 1:nVal
        nPower = mean(abs(H_val(:, :, q)).^2, 'all') / snrTest;
        HDL_val_n(:, :, q) = H_val(:, :, q) + sqrt(nPower / 2) * (randn(na, nc) + 1i * randn(na, nc));
    end
    Lambda = squeeze(1 ./ mean(abs(HDL_val_n).^2, [1 2]));
end
HDL_val_n = bsxfun(@times, HDL_val_n, reshape(sqrt(Lambda), 1, 1, [])); % nhân từng phần tử của HDL_val_n với mảng sqrt(Lambda) đã được định dạng lại thành kích thước (1,1,nVal)
HDL_val_compl_tmp = reshape(HDL_val_n, na * nc, nVal).'; % đổi hình dạng của HDL_val_n từ kích thước (na, nc, nVal) thành ma trận 2D có kích thước (nVal, na*nc)
HDL_val_compl = bsxfun(@minus, HDL_val_compl_tmp, HUL_train_compl_tmp_mean); % trừ đi giá trị trung bình tương ứng của các cột trong HUL_train_compl_tmp_mean

% DL Testing
HDL_test_n = H_test;
Lambda = squeeze(1 ./ mean(abs(HDL_test_n).^2, [1 2])); % tính nghịch đảo của giá trị trung bình bình phương phần tuyệt đối của HDL_test_n qua các chiều 1 và 2
HDL_test = bsxfun(@times, HDL_test_n, reshape(sqrt(Lambda), 1, 1, [])); % nhân từng phần tử của HDL_test_n với mảng sqrt(Lambda) đã được định hình lại thành kích thước (1,1,nTest)
if snrTest ~= -1 % MUC DICH: thêm nhiễu Gaussian vào ma trận H_test để tạo ra HDL_test_n với công suất nhiễu dựa trên snrTest
    for q = 1:nTest
        nPower = mean(abs(H_test(:, :, q)).^2, 'all') / snrTest;
        HDL_test_n(:, :, q) = H_test(:, :, q) + sqrt(nPower / 2) * (randn(na, nc) + 1i * randn(na, nc));
    end
    Lambda = squeeze(1 ./ mean(abs(HDL_test_n).^2, [1 2]));
end
HDL_test_n = bsxfun(@times, HDL_test_n, reshape(sqrt(Lambda), 1, 1, [])); % nhân từng phần tử của HDL_test_n với mảng sqrt(Lambda) đã được định dạng lại thành kích thước (1,1,nTest)
HDL_test_compl_tmp = reshape(HDL_test_n, na * nc, nTest).'; % đổi hình dạng của HDL_test_n từ kích thước (na, nc, nTest) thành ma trận 2D có kích thước (nTest, na*nc)
HDL_test_compl = bsxfun(@minus, HDL_test_compl_tmp, HUL_train_compl_tmp_mean); % trừ đi giá trị trung bình tương ứng của các cột trong HUL_train_compl_tmp_mean

%% Reshape and split real and imaginary parts
HUL_train_compl = reshape(HUL_train_compl, nTrain, na, nc);
HDL_val_compl = reshape(HDL_val_compl, nVal, na, nc);
HDL_test_compl = reshape(HDL_test_compl, nTest, na, nc);

% Split real and imaginary parts
H_train_real = split_real_image(HUL_train_compl);
H_val_real = split_real_image(HDL_val_compl);
H_test_real = split_real_image(HDL_test_compl);

% SAVE
% Save H_train_real
save('data/H_train_real.mat', 'H_train_real', '-v7.3');
% Save H_val_real
save('data/H_val_real.mat', 'H_val_real', '-v7.3');
% Save H_test_real
save('data/H_test_real.mat', 'H_test_real', '-v7.3');
% Save HDL_test
save('data/HDL_test.mat', 'HDL_test', '-v7.3');
% Save HUL_train_compl_tmp_mean
save('data/HUL_train_compl_tmp_mean.mat', 'HUL_train_compl_tmp_mean', '-v7.3');


%% Function to split real and imaginary parts
function H_normalized = split_real_image(H)
    H_real = real(H);
    H_imag = imag(H);
    H_normalized = cat(4, H_real, H_imag); % concatenate along the 4th dimension
end
