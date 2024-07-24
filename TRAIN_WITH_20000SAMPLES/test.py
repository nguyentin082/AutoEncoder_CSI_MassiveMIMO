import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.models import Model, load_model
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import h5py

nTest = 5000
na = 64
nc = 160

# Sử dụng MirroredStrategy để huấn luyện trên nhiều GPU
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Đăng ký hàm 'mse' và 'accuracy' để sử dụng khi load mô hình
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def accuracy(y_true, y_pred):
    return tf.keras.metrics.Accuracy()(y_true, y_pred)

# Đặt tất cả các bước tạo mô hình, compile và huấn luyện bên trong scope của strategy
with strategy.scope():

    # Load test data and mean normalized data
    with h5py.File('data/H_test_real.mat', 'r') as file:
        H_test_real = file['H_test_real'][:].transpose(3, 2, 1, 0)
        print("H_test_real Size: ", H_test_real.shape)

    with h5py.File('data/HUL_train_compl_tmp_mean.mat', 'r') as file:
        HUL_train_compl_tmp_mean = file['HUL_train_compl_tmp_mean'][:]
        print("HUL_train_compl_tmp_mean Size: ", HUL_train_compl_tmp_mean.shape)

    with h5py.File('data/HDL_test.mat', 'r') as file:
        HDL_test = file['HDL_test'][:]
        print("HDL_test Size: ", HDL_test.shape)

    # Define custom PSNR metric
    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)

    # Tải mô hình autoencoder
    autoencoder_model = load_model('model/autoencoder_model.h5',
                                   custom_objects={'psnr': psnr, 'mse': mse, 'accuracy': accuracy})

    autoencoder_model.summary()

    # Đảm bảo dữ liệu đầu vào có dạng float32
    H_test_real = H_test_real.astype('float32')

    # PREDICT
    print("\nPredicting.....................")

    def quantize(codeword, bits):
        max_val = np.max(np.abs(codeword))
        scale = (2 ** (bits - 1) - 1) / max_val
        quantized_codeword = np.round(codeword * scale).astype(np.int32)
        return quantized_codeword

    isQuantize = True
    if isQuantize:
        # PREDICT WITH ENCODER
        codeword = autoencoder_model.get_layer('encoder').predict(H_test_real)

        # QUANTIZE CODEWORD
        num_bits = 16
        quantized_codeword = quantize(codeword, num_bits)

        # DEQUANTIZE CODEWORD
        scale = (2 ** (num_bits - 1) - 1) / np.max(np.abs(codeword))
        dequantized_codeword = quantized_codeword.astype(np.float32) / scale

        # PREDICT WITH DECODER
        H_test_real_predict = autoencoder_model.get_layer('decoder').predict(dequantized_codeword)
        sio.savemat(f'result/testing/H_test_real_predict{num_bits}bit.mat',
                    {'H_test_real_predict': H_test_real_predict})
        print("Data successfully saved!")

        # PLOT
        sample_index = 100  # Chọn một mẫu để hiển thị, bạn có thể thay đổi chỉ số này để xem các mẫu khác
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(H_test_real[sample_index, :, :, 0], cmap='gray')
        plt.title('Input Image')
        plt.subplot(1, 2, 2)
        plt.imshow(H_test_real_predict[sample_index, :, :, 0], cmap='gray')
        plt.title('Output Image')
        plt.savefig(f'result/testing/comparison_input_output_{num_bits}bit.png')
        plt.show()
    else:
        H_test_real_predict = autoencoder_model.predict(H_test_real)
        sio.savemat(f'result/testing/H_test_real_predict_nonquant.mat',
                    {'H_test_real_predict': H_test_real_predict})
        print("Data successfully saved!")

        sample_index = 100  # Chọn một mẫu để hiển thị, bạn có thể thay đổi chỉ số này để xem các mẫu khác
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(H_test_real[sample_index, :, :, 0], cmap='gray')
        plt.title('Input Image')
        plt.subplot(1, 2, 2)
        plt.imshow(H_test_real_predict[sample_index, :, :, 0], cmap='gray')
        plt.title('Output Image')
        plt.savefig(f'result/testing/comparison_input_output_nonquant.png')
        plt.show()

    print(f"H_test_real_predict shape: {H_test_real_predict.shape}")
    print(H_test_real_predict[0, 0, 0, 0])

    # TODO: SAVE TEST PREDICT DATA
