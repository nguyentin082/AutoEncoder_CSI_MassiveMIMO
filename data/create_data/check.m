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

% Print the first element of each matrix
fprintf('First element of data1: %f\n', data1(1,1,1));
fprintf('First element of data2: %f\n', data2(1,1,1));
fprintf('First element of data3: %f\n', data3(1,1,1));
fprintf('First element of data4: %f\n', data4(1,1,1));
fprintf('First element of data5: %f\n', data5(1,1,1));
fprintf('First element of data6: %f\n', data6(1,1,1));
