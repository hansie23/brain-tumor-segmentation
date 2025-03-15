from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.models import Model

def unet_model(input_shape):
    inputs = Input(input_shape)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
    # Mid-level
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # drop3 = Dropout(0.5)(conv3)
        
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D((2, 2))(drop4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)
    # pool5 = MaxPooling2D((2, 2))(drop5)
    
    # Decoder
    upconv6 = UpSampling2D(size=(2, 2))(drop5)
    upconv6 = Conv2D(512, 2, activation='relu', padding='same')(upconv6)
    merge6 = concatenate([drop4, upconv6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    drop6 = Dropout(0.5)(conv6)
    
    upconv7 = UpSampling2D(size=(2, 2))(drop6)
    upconv7 = Conv2D(256, 2, activation='relu', padding='same')(upconv7)
    merge7 = concatenate([conv3, upconv7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    drop7 = Dropout(0.5)(conv7)
    
    upconv8 = UpSampling2D(size=(2, 2))(drop7)
    upconv8 = Conv2D(128, 2, activation='relu', padding='same')(upconv8)
    merge8 = concatenate([conv2, upconv8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    drop8 = Dropout(0.5)(conv8)
    
    upconv9 = UpSampling2D(size=(2, 2))(drop8)
    upconv9 = Conv2D(64, 2, activation='relu', padding='same')(upconv9)
    merge9 = concatenate([conv1, upconv9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    drop9 = Dropout(0.5)(conv9)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(drop9)
    
    model = Model(inputs=inputs, outputs=outputs, name='Custom-UNet')
    return model
