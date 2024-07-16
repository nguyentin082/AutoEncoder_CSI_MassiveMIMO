clc; clear; close all;

% Load the five data files (64x160x10000 matrix)
data1 = load('one.mat');
data2 = load('two.mat');
data3 = load('three.mat');
data4 = load('four.mat');
data5 = load('five.mat');
data6 = load('six.mat');

% Assuming the data in the files are stored in variables named 'H'
% If the variable names are different, adjust the field names accordingly
data1 = data1.H;
data2 = data2.H;
data3 = data3.H;
data4 = data4.H;
data5 = data5.H;
data6 = data6.H;

% Ghép các tập dữ liệu lại với nhau
combined_data = cat(3, data1, data2, data3, data4, data5, data6);

% Kiểm tra kích thước của tập dữ liệu sau khi ghép
disp(size(combined_data));  % Kết quả sẽ là [64, 160, 60000]

% Tách phần thực và phần ảo
real_part = real(combined_data);
imag_part = imag(combined_data);

% Lưu phần thực và phần ảo vào file .mat với MAT-file version 7.3
save('combined_data.mat', 'real_part', 'imag_part', '-v7.3');
