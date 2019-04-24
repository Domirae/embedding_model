from keras import models
from keras import layers
from keras import optimizers
from keras import losses
import matplotlib.pyplot as plt
from makeFastTextEm import FastEM
from data_process_em_glove import processing
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences


if __name__ == '__main__':

    base_path = '/home/ailsb/PycharmProjects/test/data'

    fns_train = [base_path + '/train/train.sent_data.txt', base_path + '/train/ratings_train.txt']
    fns_test = [base_path + '/test/test.sent_data.txt', base_path + '/test/ratings_test.txt']

    tokenizer,vocab_size, max_length, train_x, train_y, test_x, test_y = processing()
    embedding_dim =100

    fast = FastEM(max_length,vocab_size,embedding_dim)

    # train_targets , train_texts = w2v.load_data(fns_train)
    # test_targets , test_texts = w2v.load_data(fns_test)
    #
    # train_data , train_label , word_index = w2v.tokenize(train_targets,train_texts)
    # test_data , test_label,word_index2 = w2v.tokenize(test_targets,test_texts)

    embedding = fast.Convert2Vec(tokenizer)

    model = models.Sequential()
    model.add(layers.Embedding(vocab_size,embedding_dim,weights=[embedding],input_length=max_length,trainable=False))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer=optimizers.RMSprop(lr=0.001),
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])

    history = model.fit(train_x, train_y, epochs = 20, batch_size=128, validation_split = 0.2)
    results = model.evaluate(test_x, test_y)
    print(results)
    print(test_x[0:1])

    input_review = ["액션신은 확실히 괜찮은데 스토리전개가 전작에비해 너무 뻔하고 뒤로갈수록 지루해짐..차라리 앳지오브투모로우가 스토리상 훨신 난듯.."]


    review_tokens = tokenizer.texts_to_sequences(input_review)
    print(review_tokens)

    review_pad = pad_sequences(review_tokens, maxlen=43,padding='post')
    print(review_pad)

    data = np.asarray(review_pad)

    score = float(model.predict(data))

    if (score > 0.5):
        print("review는 {:.2f}% 확률로 긍정 리뷰이지 않을까 추측해봅니다.\n".format(score * 100))
    else:
        print("review는 {:.2f}% 확률로 부정 리뷰이지 않을까 추측해봅니다.\n".format((1 - score) * 100))

    model.save('/home/ailsb/PycharmProjects/test/fastText_dir/data/sentimental_model_fastText.h5')

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