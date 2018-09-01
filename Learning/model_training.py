""" Construct CNN Model for training"""
from __future__ import print_function
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import plot_model, vis_utils
import time
import glob
import csv
import itertools
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from keras import backend as K


current_dir = os.path.dirname(__file__)
parent_dir = os.path.split(current_dir)[0]
parent_dir = os.path.split(parent_dir)[0] + '/data/'
save_dir = parent_dir + 'models/'


def plot_model_history(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # summarize history for accuracy
    axs[0].plot(range(1, len(model_history.history['acc']) + 1), model_history.history['acc'])
    axs[0].plot(range(1, len(model_history.history['val_acc']) + 1), model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1, len(model_history.history['acc']) + 1), len(model_history.history['acc']) / 10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
    axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss']) / 10)
    axs[1].legend(['train', 'val'], loc='best')


def get_confusion_matrix_one_hot(model_results, truth):
    '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    assert model_results.shape == truth.shape
    num_outputs = truth.shape[1]
    confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
    predictions = np.argmax(model_results,axis=1)
    assert len(predictions)==truth.shape[0]

    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:,actual_class]==1
        prediction_for_this_class = predictions[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class==predicted_class)
            confusion_matrix[actual_class, predicted_class] = count
    assert np.sum(confusion_matrix)==len(truth)
    assert np.sum(confusion_matrix)==np.sum(truth)
    return confusion_matrix.astype(int)


def get_confusion_matrix_one_hot_top2(model_results, truth):
    '''model_results and truth should be for one-hot format, i.e, have >= 2 columns,
    where truth is 0/1, and max along each row of model_results is model result
    '''
    assert model_results.shape == truth.shape
    num_outputs = truth.shape[1]
    confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)

    for i in np.arange(0, len(model_results)):
        model_results[i][np.argmax(model_results[i], axis=0)] = 0

    predictions = np.argmax(model_results,axis=1)
    assert len(predictions)==truth.shape[0]

    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:,actual_class]==1
        prediction_for_this_class = predictions[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class==predicted_class)
            confusion_matrix[actual_class, predicted_class] = count
    assert np.sum(confusion_matrix)==len(truth)
    assert np.sum(confusion_matrix)==np.sum(truth)
    return confusion_matrix.astype(int)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def cnn_training(x_train, y_train, x_valid, y_valid, num_classes, epochs, lr, target_user):

    report = open(save_dir + target_user + "_report.txt", "w")

    """ ********************** Basic Training Configuration ************************* """
    batch_size = 32
    data_augmentation = False
    model_name = target_user + '_keras_model.h5'

    """*************************** START TRAINING*******************************"""
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_valid.shape[0], 'test samples')
    print(x_train.shape[1:])

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # # added layer
    # model.add(Conv2D(128, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))  # dimensionality of the output space.
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=lr, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    # x_train /= 255
    # x_test /= 255

#    plot_model(model, to_file='model.png')

    model_history = []
    start = time.time()

    if not data_augmentation:
        print('Not using data augmentation.')

        model_history = model_history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_valid, y_valid),
                  shuffle=True)

    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        model_history = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            epochs=epochs,
                            validation_data=(x_valid, y_valid),
                            workers=4)

    end = time.time()

    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('******************** ' + target_user + '***************************')
    print('Saved trained model at %s ' % model_path)

    plot_model_history(model_history)
    print("Model took %0.2f minutes for training." % ((end - start)/60))

    # Score trained model.
    scores = model.evaluate(x_valid, y_valid, verbose=1)
    print('Validation loss: %f, Validation accuracy: %f\n' % (scores[0], scores[1]))

    # write result to report.txt
    report.writelines("************ " + target_user + " ******************\n")
    report.writelines("Model took %0.2f minutes for training.\n" % ((end - start) / 60))
    report.writelines('Validation loss: %f, Validation accuracy: %f\n' % (scores[0], scores[1]))

    report.close()


def cnn_test(x_test, y_test, user_list, use_model):
    """************************ LOAD AND TEST EXISTING MODELS****************************"""
    # Load the trained model
    model_path = os.path.join(save_dir, use_model + '_keras_model.h5')
    model = load_model(model_path)

    model.summary()
    # # define model from base model for feature extraction from fc2 layer
    # model_fc2 = Model(input=model.input, output=model.get_layer('dense_2').output)
    # model_fc2.summary()

    # Score trained model.
    predictions = model.predict(x_test)
    # Convert class vectors to binary class matrices.
    y_test = keras.utils.to_categorical(y_test, len(user_list))

    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss: %f, test accuracy: %f' % (scores[0], scores[1]))

    confusion_matrix = get_confusion_matrix_one_hot(predictions, y_test)
    np.savetxt(save_dir + use_model + '_confuse.txt', confusion_matrix, fmt='%4d')

    # write confusion matrix to csv file
    with open(save_dir + use_model + '_confuse.csv', 'w') as file:
        writer = csv.writer(file, delimiter=',', escapechar=' ', quoting=csv.QUOTE_NONE)

        names = ""
        for n in user_list:
            names = names + ',' + user_list[n].split("\n")[0]
        writer.writerow([names])    # write headers

        for i in np.arange(0, confusion_matrix.shape[0]):
            row = str(user_list[i].split("\n")[0])
            for cnt in confusion_matrix[i]:
                row = row + ',' + str(cnt)
            writer.writerow([row])

    # plot confusion matrix
    user_names = []
    for n in user_list:
        user_names.append(user_list[n].split("\n")[0])
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=user_names, normalize=False, title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(confusion_matrix, classes=user_names, normalize=True, title='Normalized Confusion Matrix')

    confusion_matrix_top2 = get_confusion_matrix_one_hot_top2(predictions, y_test)
    plt.figure()
    plot_confusion_matrix(confusion_matrix_top2, classes=user_names, normalize=False, title='Confusion Matrix')
    plt.figure()
    plot_confusion_matrix(confusion_matrix_top2, classes=user_names, normalize=True, title='Normalized Confusion Matrix')


def final_training(x_train_cnn, y_train_cnn):
    # Load the trained model
    model = load_model(save_dir + 'multi_keras_model.h5')
    model_fc2 = Model(input=model.input, output=model.get_layer('dropout_3').output)

    cnn_features = model_fc2.predict(x_train_cnn)

    X = normalize(cnn_features, norm='l2', axis=1, copy=True, return_norm=False)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y_train_cnn, test_size=0.2, random_state=7)
    print("Loading data done!")


    scoring = 'accuracy'
    """Build Models: Let’s evaluate 6 different algorithms,
    Logistic Regression (LR)
    Linear Discriminant Analysis (LDA)
    K-Nearest Neighbors (KNN).
    Classification and Regression Trees (CART).
    Gaussian Naive Bayes (NB).
    Support Vector Machines (SVM)."""
    models = []
    # models.append(('LR', LogisticRegression()))
    # models.append(('LDA', LinearDiscriminantAnalysis()))
    # models.append(('KNN', KNeighborsClassifier()))
    # models.append(('CART', DecisionTreeClassifier()))
    # models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(C=100)))
    models.append(('NN', MLPClassifier(random_state=0,
                                       hidden_layer_sizes=[100, 100])))

    '''Evaluate each model in turn'''
    print("----------------model training accuracy--------------")
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
    # Compare algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)

    print("---------------model validation accuracy-------------")
    for name, model in models:
        model = model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        print("%s: %f" % (name, accuracy_score(y_test, predictions)))
        joblib.dump(model, save_dir + name + '.pkl')
        print("Confusion matrix:")
        print(confusion_matrix(y_test, predictions))
        print("classification report:")
        print(classification_report(y_test, predictions))


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def keras_to_tf(use_model):
    model_path = os.path.join(save_dir, use_model + '_keras_model.h5')
    model = load_model(model_path)
    frozen_graph = freeze_session(K.get_session(), output_names=[model.output.op.name])
    tf.train.write_graph(frozen_graph, save_dir, "tf_model.pb", as_text=False)

    model.summary()

    # model_fc2 = Model(input=model.input, output=model.get_layer('dropout_3').output)
    # frozen_graph = freeze_session(K.get_session(), output_names=[model.output.op.name])
    # tf.train.write_graph(frozen_graph, save_dir, "tf_model.pb", as_text=False)