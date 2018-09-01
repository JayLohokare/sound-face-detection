""" Make predict on new data, and evaluate the performance """

from src.feature_extract.data_preprocessing import *
from src.feature_extract.model_training import *

""" ********************* Configuration ********************** """
target_user = 'bing'     # use 'multi' for multi user classification

""" ****************** Load Data *************************** """
user_dic = load_user_list()

y_test, x_test = load_test_data(user=target_user, user_dic=user_dic)

""" ***************** Test the Model *********************** """
cnn_test(x_test, y_test, user_list=user_dic, use_model='multi')

plt.show()