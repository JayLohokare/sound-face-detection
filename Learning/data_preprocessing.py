# Fuse data from all user data
import glob
import os
import numpy as np
import librosa
import scipy.signal
import scipy.signal
from sklearn import model_selection


current_dir = os.path.dirname(__file__)
parent_dir = os.path.split(current_dir)[0]
parent_dir = os.path.split(parent_dir)[0] + '/data/'
data_dir = parent_dir + 'raw_data/'
out_dir = parent_dir + 'processed_data/'
test_dir = parent_dir + 'raw_test_data/'

# pre-process data for multi-class classification
def mix_data_multi_class():
    user_name = out_dir + 'multi_mixed.txt'
    user_list = parent_dir + 'models/user_list.txt'
    output_file = open(user_name, 'w')
    list_file = open(user_list, "w")

    index = 0
    for file in glob.glob(os.path.join(data_dir, "*.txt")):
        user_data = open(file, 'r')
        name_dir = user_data.name.split("/")
        namestr = name_dir[len(name_dir) - 1].split(".")[0]
        if namestr != 'nobody':
            for row in user_data:
                if row[0] == "1":
                    newstr = str(index) + row[1:]
                    output_file.writelines(newstr)
        else:
            for row in user_data:
                newstr = str(index) + row[1:]
                output_file.writelines(newstr)
        user_data.close()
        list = str(index) + ": " + namestr
        list_file.writelines(list + "\n")
        print(list)
        index = index + 1

    output_file.close()
    list_file.close()
    print("There are %d classes." % index)
    return index


# pre-process data for binary classification
def mix_data_single_class(target_user):
    user_name = out_dir + target_user + '_mixed.txt'
    output_file = open(user_name, 'w')
    negative = 0
    positive = 0

    for file in glob.glob(os.path.join(data_dir, "*.txt")):
        strs = file.split("/")
        if strs[len(strs)-1].split(".")[0] != target_user:
            user_data = open(file, 'r')
            for row in user_data:
                newstr = "0" + row[1:]
                output_file.writelines(newstr)
                negative = negative + 1
        else:
            user_data = open(file, 'r')
            for row in user_data:
                if row[0] == '1':
                    output_file.writelines(row)
                    positive = positive + 1
        user_data.close()

    output_file.close()
    print('positive: {}, negative: {}'.format(positive, negative))


# get the spectrogram
def extract(file_name):
    labels = []
    specs = []
    f = open(out_dir + file_name + '_mixed.txt', 'r')

    for row in f:
        landmarks = []
        acoustic = []
        strs = row.split(' ')
        labels.append(int(strs[0]))

        for i in np.arange(1, 17):
            landmarks.append(float(strs[i].split(':')[1]))
        for i in np.arange(17, len(strs)):
            acoustic.append(float(strs[i].split(':')[1]))
        x = np.array(acoustic[0:])
        stft = librosa.stft(x, n_fft=64, hop_length=3, window=scipy.signal.hanning, pad_mode='reflect')
        D = librosa.amplitude_to_db(stft, ref=np.max)
        specs.append(D)

    return np.array(labels), np.expand_dims(np.array(specs), axis=3)


# shuffle data
def dump_data(X, y, name):
    x_train, x_rest, y_train, y_rest = model_selection.train_test_split(X, y, test_size=0.3, random_state=7)
    x_valid, x_test, y_valid, y_test = model_selection.train_test_split(x_rest, y_rest, test_size=0.5, random_state=7)
    outfile = out_dir + name + '_mixed'
    np.save(outfile, [x_train, y_train, x_valid, y_valid, x_test, y_test])


# load data
def load_data(name):
    return np.load(out_dir + name + '_mixed.npy')


# load user list
def load_user_list():
    dic = {}
    with open(parent_dir + 'models/user_list.txt', 'r') as list:
        for row in list:
            n = int(row.split(':')[0])
            name = row.split(':')[1][1:]
            name = name[0:(len(name)-1)]
            dic[n] = name
    return dic


# load testing data
def load_test_data(user, user_dic):
    user_id = -1
    for n, name in user_dic.items():
        if name == user:
            user_id = n

    if user_id == -1:
        print('No such user!')
        quit()

    labels = []
    specs = []
    f = open(test_dir + user + '.txt', 'r')

    for row in f:
        landmarks = []
        acoustic = []
        strs = row.split(' ')
        if strs[0] == '1':
            labels.append(user_id)

            for i in np.arange(1, 17):
                landmarks.append(float(strs[i].split(':')[1]))
            for i in np.arange(17, len(strs)):
                acoustic.append(float(strs[i].split(':')[1]))
            x = np.array(acoustic[0:])
            stft = librosa.stft(x, n_fft=64, hop_length=3, window=scipy.signal.hanning, pad_mode='reflect')
            D = librosa.amplitude_to_db(stft, ref=np.max)
            specs.append(D)

    return np.array(labels), np.expand_dims(np.array(specs), axis=3)
