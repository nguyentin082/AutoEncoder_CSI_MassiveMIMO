import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio

# Sử dụng MirroredStrategy để huấn luyện trên nhiều GPU
strategy = tf.distribute.MirroredStrategy()

print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Đặt tất cả các bước tạo mô hình, compile và huấn luyện bên trong scope của strategy
with strategy.scope():

    # Encoder model as per the paper specifications
    def encoder(input_shape=(64, 160, 2), latent_dim=256):
        input_layer = Input(shape=input_shape)
        x = Conv2D(8, (3, 3), strides=(2, 2), padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(16, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Flatten()(x)
        x = Dense(latent_dim)(x)
        x = Activation('tanh')(x)

        model = Model(inputs=input_layer, outputs=x, name='encoder')
        return model


    # Decoder model as per the paper specifications
    def decoder(latent_dim=256):
        input_layer = Input(shape=(latent_dim,))
        x = Dense(1280)(input_layer)
        x = Reshape((2, 5, 128))(x)

        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2DTranspose(8, (3, 3), strides=(2, 2), padding='same')(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)

        x = Conv2DTranspose(2, (3, 3), strides=(1, 1), padding='same')(x)

        model = Model(inputs=input_layer, outputs=x, name='decoder')
        return model


    # Autoencoder model combining encoder and decoder
    def autoencoder(input_shape=(64, 160, 2), latent_dim=256):
        encoder_model = encoder(input_shape, latent_dim)
        decoder_model = decoder(latent_dim)

        input_layer = Input(shape=input_shape)
        encoded = encoder_model(input_layer)
        decoded = decoder_model(encoded)

        model = Model(inputs=input_layer, outputs=decoded, name='autoencoder')
        return model


    # Định nghĩa mô hình encoder
    encoder_model = encoder(input_shape=(64, 160, 2), latent_dim=256)
    encoder_model.summary()

    # Định nghĩa mô hình decoder
    decoder_model = decoder(latent_dim=256)
    decoder_model.summary()

    # Định nghĩa mô hình autoencoder
    autoencoder_model = autoencoder(input_shape=(64, 160, 2), latent_dim=256)
    autoencoder_model.summary()

    # Tính tổng số lượng tham số của model
    encoder_params = encoder_model.count_params()
    decoder_params = decoder_model.count_params()

    print(f"Tổng số lượng tham số của mô hình encoder: {encoder_params}")
    print(f"Tổng số lượng tham số của mô hình decoder: {decoder_params}")

    # Tạo tên tệp với thông tin kích thước
    encoder_filename = f'encoder_params_dim256.mat'
    decoder_filename = f'decoder_params_dim256.mat'

    # Lưu dữ liệu vào file .mat
    sio.savemat(encoder_filename, {'encoder_params': encoder_params})
    sio.savemat(decoder_filename, {'decoder_params': decoder_params})

    print(f"Dữ liệu của encoder đã được lưu vào file {encoder_filename}")
    print(f"Dữ liệu của decoder đã được lưu vào file {decoder_filename}")


