import utils
from utils import *

from tqdm import tqdm 
import psutil
from sys import getsizeof
from kfold_rare_class import * 
import os 




if __name__ == "__main__":


	K.set_image_data_format('channels_first')
	K.image_data_format()



	df_train = pd.read_csv('data/train_v2.csv')
	df_test = pd.read_csv('data/sample_submission_v2.csv')

	flatten = lambda l: [item for sublist in l for item in sublist]
	labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

	labels = ['blow_down',
	 'bare_ground',
	 'conventional_mine',
	 'blooming',
	 'cultivation',
	 'artisinal_mine',
	 'haze',
	 'primary',
	 'slash_burn',
	 'habitation',
	 'clear',
	 'road',
	 'selective_logging',
	 'partly_cloudy',
	 'agriculture',
	 'water',
	 'cloudy']

	label_map = {'agriculture': 14,
	 'artisinal_mine': 5,
	 'bare_ground': 1,
	 'blooming': 3,
	 'blow_down': 0,
	 'clear': 10,
	 'cloudy': 16,
	 'conventional_mine': 2,
	 'cultivation': 4,
	 'habitation': 9,
	 'haze': 6,
	 'partly_cloudy': 13,
	 'primary': 7,
	 'road': 11,
	 'selective_logging': 12,
	 'slash_burn': 8,
	 'water': 15}


	# 'primary', 'agriculture','water', 'cultivation', "habitation"
	#add some labels to the list
	rare_labels = [] # list(set(labels) - set(['clear','cloudy','haze','partly_cloudy']))




	for class_name in rare_labels:

		if not os.path.exists("data/cache/{}".format(class_name)):
			os.makedirs("data/cache/{}".format(class_name))

		if not os.path.exists("weights/{}".format(class_name)):
			os.makedirs("weights/{}".format(class_name))

		print "starting ops for class ", class_name
		train_and_predict_for_class(class_name)