"""
IBM Deep Learning (IDE) Generated Code.
Compatible Keras Version : 2.1
Tested on Python Version : 3.6.3
"""

# Import all dependencies
import os
import numpy as np
import keras
from keras.models import Model
import keras.backend as K
import keras.regularizers as R
import keras.constraints as C
from keras.layers import *
from keras.optimizers import *
import keras_helper as helper
from keras.callbacks import TensorBoard

# Perform data pre-processing
defined_metrics = []
batch_size = 256
num_epochs = 10

ImageData_1_params = {
    "train_dataset": "train.p",
    "val_dataset": "",
    "test_dataset": "test.p",
    "validation_split": 0.1,
    "test_split": 0.1,
    "rows": 32,
    "cols": 32,
    "dim_ordering": "channels_last",
    "dbformat": "Python Pickle",
    "num_classes": 43
}
ImageData_1_data = helper.image_data_handler(ImageData_1_params)
train_x = ImageData_1_data["train_x"]
train_y = ImageData_1_data["train_y"]
val_x = ImageData_1_data["val_x"]
val_y = ImageData_1_data["val_y"]
test_x = ImageData_1_data["test_x"]
test_y = ImageData_1_data["test_y"]
labels = ImageData_1_data["labels"]
ImageData_1_shape = train_x.shape[1:]

# Define network architecture
ImageData_1 = Input(shape=ImageData_1_shape)
Convolution2D_2 = Conv2D(
    32, (3, 3),
    data_format="channels_last",
    strides=(1, 1),
    padding="valid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="glorot_uniform",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None)(ImageData_1)
ReLU_13 = Activation("relu")(Convolution2D_2)
Pooling2D_4 = MaxPooling2D(
    pool_size=(2, 2),
    strides=(1, 1),
    padding="valid",
    data_format="channels_last")(ReLU_13)
Convolution2D_3 = Conv2D(
    64, (3, 3),
    data_format="channels_last",
    strides=(1, 1),
    padding="valid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="glorot_uniform",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None)(Pooling2D_4)
ReLU_14 = Activation("relu")(Convolution2D_3)
Pooling2D_5 = MaxPooling2D(
    pool_size=(2, 2),
    strides=(1, 1),
    padding="valid",
    data_format="channels_last")(ReLU_14)
Dropout_6 = Dropout(0.25)(Pooling2D_5)
Flatten_7 = Flatten()(Dropout_6)
Dense_8 = Dense(
    128,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="glorot_uniform",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None)(Flatten_7)
ReLU_15 = Activation("relu")(Dense_8)
Dropout_9 = Dropout(0.5)(ReLU_15)
Dense_10 = Dense(
    43,
    use_bias=True,
    kernel_initializer="glorot_uniform",
    bias_initializer="glorot_uniform",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None)(Dropout_9)
Sigmoid_16 = Activation("sigmoid")(Dense_10)
defined_loss = "categorical_crossentropy"
defined_metrics.append("accuracy")

model_inputs = [ImageData_1]
model_outputs = [Sigmoid_16]
model = Model(inputs=model_inputs, outputs=model_outputs)

# Define optimizer
learning_rate = 0.100000
decay = 0.100000
beta_1 = 0.900000
beta_2 = 0.999000
optim = Adam(lr=learning_rate, beta_1=beta_1, beta_2=beta_2, decay=decay)

# Perform training and other misc. final steps
model.compile(loss=defined_loss, optimizer=optim, metrics=defined_metrics)
if len(model_outputs) > 1:
    train_y = [train_y] * len(model_outputs)
    if len(val_x) > 0: val_y = [val_y] * len(model_outputs)
    if len(test_x) > 0: test_y = [test_y] * len(model_outputs)
#writing metrics
job_state_dir = os.environ.get('JOB_STATE_DIR')
static_path = os.path.join("logs", "tb", "test")
if job_state_dir is not None:
    tb_directory = os.path.join(job_state_dir, static_path)
else:
    tb_directory = static_path
tensorboard = TensorBoard(log_dir=tb_directory)
if (len(val_x) > 0):
    model.fit(
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_data=(val_x, val_y),
        shuffle=True,
        callbacks=[tensorboard])
else:
    model.fit(
        train_x,
        train_y,
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=1,
        shuffle=True,
        callbacks=[tensorboard])
if (len(test_x) > 0):
    test_scores = model.evaluate(test_x, test_y, verbose=1)
    print(test_scores)
if "model_result_path" not in locals() \
 and "model_result_path" not in globals():
    model_result_path = "./keras_model.hdf5"
model.save(model_result_path)
print("Model saved in file: %s" % model_result_path)
