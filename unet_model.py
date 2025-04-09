from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, BatchNormalization, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU

def unet_model(input_shape=(112, 112, 3)):
    """
    A U-Net model for brain tumor segmentation, designed to handle (128, 128) images effectively.

    Args:
        input_shape: Tuple, shape of the input images (height, width, channels).

    Returns:
        model: Keras Model instance.
    """
    inputs = Input(input_shape)

    # Encoder
    # Block 1
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.1)(pool1)

    # Block 2
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.1)(pool2)

    # Block 3
    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.2)(pool3)

    # Block 4
    conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    conv4 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.2)(pool4)
    
    # Mid-level (Bottleneck)
    conv5 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    conv5 = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = LeakyReLU(alpha=0.1)(conv5)
    drop5 = Dropout(0.3)(conv5)
    
    # Decoder
    # Block 6
    upconv6 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(drop5) # Conv2DTranspose instead of upsampling
    merge6 = concatenate([conv4, upconv6], axis=3)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = LeakyReLU(alpha=0.1)(conv6)
    conv6 = Dropout(0.2)(conv6)

    # Block 7
    upconv7 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv6)
    merge7 = concatenate([conv3, upconv7], axis=3)
    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = LeakyReLU(alpha=0.1)(conv7)
    conv7 = Dropout(0.1)(conv7)

    # Block 8
    upconv8 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv7)
    merge8 = concatenate([conv2, upconv8], axis=3)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)
    conv8 = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = LeakyReLU(alpha=0.1)(conv8)
    conv8 = Dropout(0.1)(conv8)
    
    # Block 9
    upconv9 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same', kernel_initializer='he_normal')(conv8)
    merge9 = concatenate([conv1, upconv9], axis=3)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)
    conv9 = Conv2D(32, 3, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = LeakyReLU(alpha=0.1)(conv9)


    outputs = Conv2D(3, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs, name='Custom-UNet')
    return model
