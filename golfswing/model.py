from keras import layers, Model, applications

def build_cnn_base(input_shape=(160,160, 3), trainable=False):
    """
    Base CNN for Model.
    
    Input Shape = Deminsions of videos
    """
    base_model = applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
        pooling='avg',
        )
    base_model.trainable = trainable
    
    return base_model

def build_LSTM(
        clip_len=64,
        image_size=(160, 160, 3),
        num_classes=9,
        units=128, 
        bidirectional=True, 
        dropout = .3,
        ):
    """
    LSTM Heead for Model 

    """
    input_layer = layers.Input(shape=(clip_len, *image_size), name="Clip")
    
    base = build_cnn_base(image_size, trainable=False)
    x = layers.TimeDistributed(base, name="cnn_base")(input_layer)
    
    if bidirectional:
        x = layers.Bidirectional(
            layers.LSTM(units, return_sequences=True), 
            name="Bi_Dir_LSTM"
        )(x)
    else:
        x = layers.LSTM(units, return_sequences=True, name="LSTM")(x)
        
    x = layers.Dropout(dropout)(x)
    
    out = layers.TimeDistributed(layers.Dense(num_classes, activation='softmax'), 
                                 name="Output")(x)
    
    model = Model(input_layer, out, name="LSTM_Golf")
    
    return model 

def load_trained_model(weights_path, clip_len=64, units=256, bidirectional=True):
    """
    Build the model, then load saved weights into it
    The settings must match the ones the model was trained with.
    """
    model = build_LSTM(
        clip_len=clip_len,
        units=units,
        bidirectional=bidirectional
    )
    model.load_weights(weights_path)
    return model