from keras import models
from keras import layers
from keras import optimizers
from keras import losses
import matplotlib.pyplot as plt
from bow_dir.data_process import processing
import os

os.environ["CUDA_DEVICE_ORDER"]=""

if __name__ == '__main__':

    train_x,train_y,test_x,test_y = processing()

    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(6500,)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])

    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    history = model.fit(train_x, train_y, epochs=20, batch_size=128, validation_split=0.2)
    results = model.evaluate(test_x, test_y)

    print(results)
    model.save('sentimental_model.h5')
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc,'b', label = "Validation acc")
    plt.title('Training and Validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss,'b', label = "Validation loss")
    plt.title('Training and Validation loss')
    plt.legend()

    plt.show()