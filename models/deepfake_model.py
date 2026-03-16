import tensorflow as tf
from keras import layers , models, applications
from keras.layers import TimeDistributed, LSTM, Dense

def build_models():

    base = applications.EfficientNetB5(
        include_top=False,
        weights='imagenet',
        input_shape=(224,224,3),
        pooling='avg'
    )

    base.trainable= False

    model = models.Sequential([
        TimeDistributed(base),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(1,activation='sigmoid')
    ])

    return model