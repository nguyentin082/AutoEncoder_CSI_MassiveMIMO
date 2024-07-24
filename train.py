import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
import numpy as np
import matplotlib.pyplot as plt
import h5py

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

    # Định nghĩa hàm PSNR
    def psnr(y_true, y_pred):
        return tf.image.psnr(y_true, y_pred, max_val=1.0)


    # Thiết lập optimizer với learning rate ban đầu
    initial_learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

    # Compile mô hình với optimizer và hàm loss mới
    autoencoder_model.compile(optimizer=optimizer, loss='mse', metrics=[psnr, 'accuracy'])

    # Callback giảm learning rate khi loss không cải thiện
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-4, verbose=1)

    # Callback để lưu lại lịch sử loss và accuracy
    class MetricsHistory(Callback):
        def on_train_begin(self, logs=None):
            self.losses = {'train': [], 'val': []}
            self.psnr = {'train': [], 'val': []}
            self.accuracy = {'train': [], 'val': []}

        def on_epoch_end(self, epoch, logs=None):
            self.losses['train'].append(logs.get('loss'))
            self.losses['val'].append(logs.get('val_loss'))
            self.psnr['train'].append(logs.get('psnr'))
            self.psnr['val'].append(logs.get('val_psnr'))
            self.accuracy['train'].append(logs.get('accuracy'))
            self.accuracy['val'].append(logs.get('val_accuracy'))

    history_callback = MetricsHistory()

    # LOAD NORMALIZED DATA
    # Load data from MAT files
    with h5py.File('data/H_train_real.mat', 'r') as file:
        H_train_real = np.array(file['H_train_real']).transpose(3, 2, 1, 0)
        print("\nH_train_real: ", H_train_real.shape)
        print(H_train_real[0, 0, 0, 0])
    with h5py.File('data/H_val_real.mat', 'r') as file:
        H_val_real = np.array(file['H_val_real']).transpose(3, 2, 1, 0)
        print("\nH_val_real: ", H_val_real.shape)
        print(H_val_real[0, 0, 0, 0])
    # H_train_real float(40000x64x160x2)
    # H_val_real float(5000x64x160x2)
    # # TODO: GET LESS THAN TRAINING SAMPLES
    # H_train_real = H_train_real[:5000, :, :, :]
    # print("\nH_train_real be scaled: ", H_train_real.shape)
    # print(H_train_real[0, 0, 0, 0])

    # TRAINING
    history = autoencoder_model.fit(
        H_train_real, H_train_real,
        epochs=100,
        batch_size=64,
        shuffle=True,
        validation_data=(H_val_real, H_val_real),
        callbacks=[reduce_lr, history_callback]
    )

    # Vẽ đồ thị loss
    plt.figure(figsize=(10, 5))
    plt.plot(history_callback.losses['train'], label='Train Loss')
    plt.plot(history_callback.losses['val'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('result/training/training_validation_loss.png')
    plt.show()

    # Vẽ đồ thị PSNR
    plt.figure(figsize=(10, 5))
    plt.plot(history_callback.psnr['train'], label='Train PSNR')
    plt.plot(history_callback.psnr['val'], label='Validation PSNR')
    plt.title('Training and Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    plt.savefig('result/training/training_validation_psnr.png')
    plt.show()

    # Vẽ đồ thị Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history_callback.accuracy['train'], label='Train Accuracy')
    plt.plot(history_callback.accuracy['val'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('result/training/training_validation_accuracy.png')
    plt.show()

    # Lưu mô hình autoencoder vào tệp H5
    autoencoder_model.save('model/autoencoder_model.h5')
    print("Mô hình đã được lưu thành công vào autoencoder_model.h5")

    # Dự đoán đầu ra từ dữ liệu đầu vào
    H_train_real_pred = autoencoder_model.predict(H_train_real)

    # Tính toán Mean Squared Error (MSE) giữa đầu vào và đầu ra
    mse = np.mean((H_train_real - H_train_real_pred) ** 2)
    print(f'Mean Squared Error (MSE) giữa đầu vào và đầu ra: {mse}')

    # Tính toán PSNR giữa đầu vào và đầu ra
    psnr_value = tf.image.psnr(H_train_real, H_train_real_pred, max_val=1.0)
    print(f'PSNR giữa đầu vào và đầu ra: {np.mean(psnr_value)}')

    # Vẽ hình ảnh so sánh giữa đầu vào và đầu ra
    index = np.random.randint(0, H_train_real.shape[0])  # Chọn một mẫu ngẫu nhiên để hiển thị
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(H_train_real[index, :, :, 0], cmap='gray')
    plt.title('Đầu vào')

    plt.subplot(1, 2, 2)
    plt.imshow(H_train_real_pred[index, :, :, 0], cmap='gray')
    plt.title('Đầu ra')

    # Lưu hình ảnh
    plt.savefig('result/training/comparison_input_output.png')
    plt.show()
