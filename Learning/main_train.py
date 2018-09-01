""" The main script for face authentication"""
from src.feature_extract.data_preprocessing import *
from src.feature_extract.model_training import *

""" ********************* Configuration ********************** """
target_user = 'multi'     # use 'multi' for multi user classification

""" ********************** Data Mixing *********************** """
# n_classes = mix_data_multi_class()
# mix_data_single_class(target_user)

# y, X = extract('multi')
# dump_data(X, y, 'multi')

# y, X = extract(target_user)
# dump_data(X, y, target_user)

""" ****************** Load Data *************************** """
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(target_user)
print('Training samples: {}, Validation samples: {}, Testing samples: {}'
       .format(x_train.shape[0], x_valid.shape[0], x_test.shape[0]))

user_dic = load_user_list()
n_classes = len(user_dic)

""" ***************** Train the Model ********************** """
cnn_training(x_train, y_train, x_valid, y_valid, num_classes=n_classes, epochs=100, lr=0.00001, target_user='multi')

""" ***************** Test the Model *********************** """
# cnn_test(x_test, y_test, user_list=user_dic, use_model=target_user)

""" ***************** Final Classification *********************** """
# final_training(x_valid, y_valid)

keras_to_tf(use_model=target_user)

plt.show()
