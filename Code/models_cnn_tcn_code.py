# Import librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

import keras
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Flatten, MaxPool1D , Conv1D, LayerNormalization, BatchNormalization, SpatialDropout1D
from keras.regularizers import l1, l2
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Import Keras Temporal Convolutional Network package
#pip install keras-tcn
#pip install keras-tcn --no-dependencies  # without the dependencies if you already have TF/Numpy.
from tcn import TCN, tcn_full_summary

# Import package Ordinal regression in Tensorflow Keras
# Install source package from GitHub : 
# pip install git+https://github.com/ck37/coral-ordinal/
# (or) !pip install --force-reinstall --no-deps git+https://github.com/ck37/coral-ordinal/
# or Install the stable version via pip: 
# pip install coral-ordinal
import coral_ordinal as coral
print("CORAL Ordinal version:", coral.__version__)


# The following code is intended to deal with the ordinal regression prediction of the coral-ordinal package.
# This is used in the main function for models running in the Prediction section. 
def coral_labels(x: tf.Tensor, conditional):
    '''
    Converts predicted probabilities from extended binary format to integer class labels.
    This can then be used to calculate other metrics like accuracy or mean absolute error.
    See : https://github.com/Raschka-research-group/corn-ordinal-neuralnet/blob/main/notebooks/03_conditional-method-code-example.ipynb
    See also : https://colab.research.google.com/drive/1AQl4XeqRRhd7l30bmgLVObKt5RFPHttn#scrollTo=2pfaK4doXvcS
    Notice that the probability distribution for each observation is unimodal, which is what we want for an ordinal outcome variable.
    conditional : 'v1', 'v2' or 'v2-argmax'
    '''
    
    threshold = 0.5
    
    #3 ways to deal with logit-based cumulative probs
    if conditional=='v1':
        #with cumprod
        x = tf.math.cumprod(x, axis=1)
        predict_levels = tf.cast(x > threshold, dtype=tf.int32)
        predicted_labels = tf.reduce_sum(predict_levels, axis=1)
        return predicted_labels

    elif conditional=='v2':
        #without cumprod
        predict_levels = tf.cast(x > threshold, dtype=tf.int32)
        predicted_labels = tf.reduce_sum(predict_levels, axis=1)
        return predicted_labels

    elif conditional=='v2-argmax':
        #calculate the probs of (y=0, y=1, y=2,…) using (1-P(y>0), P(y>0)(1-P(y>1|y>0), P(y>0)P(y>1|y>0)(1-P(y>2|y>1),…), and then using argmax to find the predicted label.
        ones = tf.ones((x.shape[0], 1))#.to(device)
        comp_1 = tf.concat((ones, tf.math.cumprod(x, axis=1)), axis=1)
        comp_2 = tf.concat((1-x, ones), axis=1)
        probas_y = tf.math.multiply(comp_1, comp_2)
        predicted_labels = tf.math.argmax(probas_y, axis=1)
        return predicted_labels
    
    else :
        raise ValueError("condition unknown.")



# Warning ! For training, validation and test set preparation, please ensure that each anatomical part is taken independently.

# Model code of the CNN, TCN run
def run_cnn_tcn_model(X_tr, y_tr, X_va, y_va, X_test, y_test, prediction_type, model, batch_size, len_sp = 17999):
    '''CNN and TCN Models running function that return the test target and the predicted test.
    Please enter X_tr (spectra), y_tr (age), X_va (spectra), y_va (age), X_test (spectra),
    y_test (age) that have been previously splited by sklearn.model_selection.StratifiedKFold 
    on specimens to avoid overfitting.
    For classification and ranking, encoding function may be needed on target feature.
    To understand ranking prediction please see : 
    https://arxiv.org/abs/2111.08851
    https://arxiv.org/abs/1901.07884
    prediction_type : 'classification', 'ranking' or 'regression',
    model : 'cnn' or 'tcn',
    batch_size = batch size,
    len_sp : spectrum length size initially set to 17999 to handle 18000 size spectra.'''

    # data preparation
    training_data = np.array([spectre[:len_sp] for spectre in X_tr.values])
    validation_data = np.array([spectre[:len_sp] for spectre in X_va.values])
    test_data = np.array([np.asarray(spectre[:len_sp]).astype('float32') for spectre in X_test.values])

    y_training_data = y_tr
    y_validation_data = y_va
    y_test_data = y_test

    X_train = training_data
    y_train = np.array(y_training_data.values.astype('float32').reshape(-1,1))
    print(f'Train data dimensions: {X_train.shape}, {y_train.shape}')

    X_va = validation_data
    y_va = np.array(y_validation_data.values.astype('float32').reshape(-1,1))
    print(f'Validation data dimensions: {X_va.shape}, {y_va.shape}')

    X_test = np.stack(test_data)
    y_test = np.array(y_test_data.values.astype('float32').reshape(-1,1))
    print(f'Test data dimensions: {X_test.shape}, {y_test.shape}')

    if prediction_type=='classification':
        y_train = to_categorical(y_train, num_classes= 3)
        y_test = to_categorical(y_va, num_classes= 3)

    # define models
    inputs = Input(shape=(X_train.shape[1], 1))

    if model == 'cnn':
        x = Conv1D(3, 6, activation='relu')(inputs)
        x = MaxPool1D(pool_size=100)(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
    
    elif model == 'tcn':
        x = BatchNormalization(momentum=0.98)(inputs)
        x = SpatialDropout1D(0.3)(x)
        x = TCN(nb_filters=3, kernel_size=10, nb_stacks=1, dilations=[10], padding='same',use_skip_connections=True,
                dropout_rate=0.3, return_sequences=True)(x)
        x = MaxPool1D(pool_size=50)(x)
        x = Flatten()(x)
        x = Dense(1024,activation=tf.keras.layers.PReLU(), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = Dense(512,activation=tf.keras.layers.PReLU(), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = Dense(1024,activation=tf.keras.layers.PReLU(), kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
    
    else :
        raise ValueError("model unknown. Choose 'cnn' or 'tcn'.")

    x = LayerNormalization()(x)

    ## output layer and model compilation
    if prediction_type=='classification':
        predictions = Dense(3, activation='softmax')(x) # for 3 classes classification
        model = Model(inputs=[inputs], outputs=predictions)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                       metrics=[tf.keras.metrics.CategoricalAccuracy()])

        # Define Early Stopping:
        early_stop = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, 
                            patience=20, verbose=1, mode='auto',
                            baseline=0, restore_best_weights=True)
    
    elif prediction_type=='ranking':
        NUM_CLASSES = 4 # for 3 classes classification (3+1)
        predictions = coral.CornOrdinal(NUM_CLASSES)(x)
        model = Model(inputs=[inputs], outputs=predictions)
        model.compile(optimizer='adam', loss = coral.OrdinalCrossEntropy(NUM_CLASSES),
                      metrics = [coral.MeanAbsoluteErrorLabels()])

        # Define Early Stopping:
        early_stop = EarlyStopping(monitor='val_mean_absolute_error_labels', min_delta=0, 
                              patience=20, verbose=1, mode='auto',
                              baseline=0, restore_best_weights=True)
        
    elif prediction_type=='regression':
        predictions = Dense(1, activation='relu')(x)
        model = Model(inputs=[inputs], outputs=predictions)
        model.compile(optimizer='adam', loss='huber_loss', metrics=['mae'])

        # Define Early Stopping:
        early_stop = EarlyStopping(monitor='val_mae', min_delta=0, 
                            patience=20, verbose=1, mode='auto',
                            baseline=0, restore_best_weights=True)
    
    else :
        raise ValueError("prediction_type unknown. Please choose 'classification', 'ranking' or 'regression'.")
    
    
    print(model.summary())
    
    # fit model
    EPOCH = 50 
    BATCH = batch_size

    History = model.fit(X_train, y_train,
                    epochs=EPOCH,
                    batch_size=BATCH,
                    validation_data=(X_va, y_va),
                    shuffle=True,verbose=1,
                    callbacks=[early_stop])


    history_dict=History.history
    loss_values = history_dict['loss']
    val_loss_values=history_dict['val_loss']
    plt.plot(loss_values,'bo',label='training_loss')
    plt.plot(val_loss_values,'r',label='val loss')
    plt.legend(loc = 'upper right')

    plt.show()

    results = model.evaluate(X_va,y_va, verbose = 1)
    results = dict(zip(model.metrics_names,results)) 
    print(results) #to see validation results

    #model prediction
    if prediction_type=='classification':
        y_test_pred = model.predict(X_test)
        y_test_pred = np.argmax(y_test_pred, axis=1)
        y_test_pred = y_test_pred.ravel()

    elif prediction_type=='ranking':
        ordinal_logits = model.predict(X_test, batch_size = 6) # Note that these are ordinal (cumulative) logits, not probabilities or regular logits.
        # Convert from logits to label probabilities. This is initially a tensorflow tensor.
        tensor_probs = tf.math.sigmoid([ordinal_logits])[0] #convert the cumulative logit output of the layer into the probability estimate for each ordinal label. 
        print(pd.DataFrame(tensor_probs.numpy())) # To convert the tensor into a pandas dataframe.
        y_test_pred = coral_labels(tensor_probs,'v1').numpy()

    elif prediction_type=='regression':
        y_test_pred = model.predict(X_test)
        y_test_pred = y_test_pred.ravel()

    else :
        raise ValueError("Prediction ! prediction_type unknown. Please choose 'classification', 'ranking' or 'regression'.")

    
    y_test = y_test.ravel()

    del model #to avoid overfitting on next train run
    
    return y_test, y_test_pred
    
    
    