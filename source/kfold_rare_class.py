import utils
from utils import *

from tqdm import tqdm 
import psutil
from sys import getsizeof
from sklearn.utils import class_weight
from keras.callbacks import CSVLogger
import os 
from sklearn.metrics import classification_report
import logging


logger = None

target_size = (100,100)#(256, 256)
image_feature_size = (256, 25, 25)





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



def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
	num_classes = 2
	def mf(x):
		p2 = np.zeros_like(p)
		for i in range(num_classes):
			p2[:, i] = (p[:, i] > x[i]).astype(np.int)
		score = fbeta_score(y, p2, beta=2, average='samples')
		return score

	x = [0.2]*num_classes
	for i in range(num_classes):
		best_i2 = 0
		best_score = 0
		for i2 in range(resolution):
			threshold = float(i2) / resolution
			x[i] = threshold
			score = mf(x)
			if score > best_score:
				best_i2 = threshold
				best_score = score

		x[i] = best_i2
		if verbose:
			print(i, best_i2, best_score)

	return x




def create_base_vgg(extend_top=False,size=(100,100)):
	model = Vgg16BN(size).model


	for i in range (15): #excluding the six 512 blocks at the top, was 15
		model.pop()

	for l in model.layers:
		l.trainable = False 
	
	
	if extend_top:
		for l in get_lrg_layers(): 
			model.add(l)
			
			
	return model

def get_lrg_layers():
	global image_feature_size
	nf=64; p=0.6

	layers =  [
		BatchNormalization(axis=1, input_shape= image_feature_size), #conv_layers[-1].output_shape[1:]),
		Convolution2D(nf,(3,3), activation='relu', padding='valid'),
		Dropout(p/2),


		BatchNormalization(axis=1),
		Convolution2D(nf,(3,3), activation='relu', padding='valid'),
		Dropout(p/2),

		BatchNormalization(axis=1),
		Convolution2D(nf,(3,3), activation='relu', padding='valid'),
		Dropout(p/2),
		
		BatchNormalization(axis=1),

		Flatten(),

		Dense(512,activation='relu'),
		BatchNormalization(),
		Dropout(p),  
		

		Dense(512,activation='relu'),
		BatchNormalization(),
		Dropout(p),  
		
		Dense(1,activation='sigmoid')]

	return layers   


def create_logger(class_name):
	global logger

	logger = logging.getLogger(__name__)
	logger.setLevel(logging.INFO)

	# create a file handler
	handler = logging.FileHandler("data/cache/{}/logs.txt".format(class_name))
	handler.setLevel(logging.INFO)

	logger.addHandler(handler)
	logger.info('Logger initiated!')


def print_and_log(message):
	print message
	logger.info(message)

def train_and_predict_for_class(class_name):


	############# prep
	global logger
	global target_size
	global image_feature_size



	create_logger(class_name)
	current_fold = 0 
	n_folds = 5
	num_augmentation_sets_per_fold = 1


	yfull_test = []
	# yfull_train =[]
	thresholds = []
	# oof_preds = np.zeros((x_train.shape[0],17))


	val_errors = []
	val_accuracies = []


	vgg_common = create_base_vgg()
	vgg_common.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])


	############# Action

	class_index = [label_map[class_name]]


	K.set_image_data_format('channels_first')
	K.image_data_format()



	df_train = pd.read_csv('data/train_v2.csv')
	df_test = pd.read_csv('data/sample_submission_v2.csv')

	flatten = lambda l: [item for sublist in l for item in sublist]
	labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

	


	y_train = load_array("data/cache/ytrain.dat")
	y_train = y_train[:,class_index].ravel()


	new_class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
	print ("new class weights", new_class_weights)

	# y_train = np.hstack(((y_train ==0 ).reshape(-1,1),y_train.reshape(-1,1)))


	train_data_memmapped = np.memmap("data/cache/xtrain_100x100.memmapped", dtype='float32', mode='r', 
                   shape=(len(y_train),3,100,100))



	################################ KFOLD ###########################3


		   
	kf = KFold( n_splits=n_folds, shuffle=True, random_state=2020)

	for train_index, valid_index in kf.split(y_train):
			# start_time_model_fitting = time.time()






		# X_train = x_train[train_index]
		X_valid = train_data_memmapped[valid_index]


		Y_train = y_train[train_index]
		Y_valid = y_train[valid_index]
		


		data_length = len(Y_train)
		


		current_fold += 1



		print('Start KFold number {} from {}'.format(current_fold, n_folds))
		logger.info('Start KFold number {} from {}'.format(current_fold, n_folds))


		# print('Split train: ', len(X_train), len(Y_train))
		# print('Split valid: ', len(X_valid), len(Y_valid))
		

		kfold_weights_path = os.path.join('weights/{}/'.format(class_name),
		 'vgg1_weights_kfold_' + str(current_fold) + '.h5')

		callbacks = [EarlyStopping(monitor='val_loss', patience=5),
					 ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True),
					 CSVLogger("data/cache/{}/keras_log_fold{}.csv".format(class_name,current_fold),append=False)]
 




		##################   TRAINING


		conv_trn_feat = np.memmap("data/cache/vgg_features_train_100x100_part{}of{}.planet".\
			format(current_fold,n_folds)
			, dtype='float32', mode='r', 
			shape=(num_augmentation_sets_per_fold*data_length,)+ image_feature_size)


		Y_train = np.concatenate([Y_train]*num_augmentation_sets_per_fold) #duplicate targets too
		


		print "create conv_valid_feat"		
		conv_valid_feat = vgg_common.predict(X_valid, batch_size=32, verbose=1)



		model =  Sequential(get_lrg_layers())
		model.compile(optimizer="nadam", loss='binary_crossentropy', metrics=['accuracy'])

		
		
		######################## ACTION
		history = model.fit(x = conv_trn_feat, y= Y_train, validation_data=(conv_valid_feat, Y_valid),
		  batch_size=32, epochs=50,callbacks=callbacks, 
		  class_weight={0:new_class_weights[0],1:new_class_weights[1]},
		  shuffle=True,verbose=1)

		
		
		######################## VALIDATION 
		
		if os.path.isfile(kfold_weights_path):
			print "loading best weights from ",kfold_weights_path
			model.load_weights(kfold_weights_path)
			
		
		## verifying local validation results
		p_valid = model.predict(conv_valid_feat, batch_size = 32, verbose=1)
		class_0_preds = 1 - p_valid.ravel()
		preds_val = np.hstack((class_0_preds.reshape(-1,1), p_valid.reshape(-1,1)))	
		target_val = np.hstack(((Y_valid.ravel()==0).reshape(-1,1),Y_valid.reshape(-1,1))).astype(np.uint8)


		local_best_thresholds = optimise_f2_thresholds(target_val, preds_val)
#         print(local_best_thresholds)
		thresholds.append(local_best_thresholds)

		
		# a bit redundant but leaving optimization out for now	
		rare_class_prediction = preds_val[:,1] > local_best_thresholds[1]
		others_prediction = rare_class_prediction == 0 
		preds_val = np.hstack((others_prediction.reshape(-1,1),rare_class_prediction.reshape(-1,1)))
		report = classification_report(target_val, preds_val.astype(np.uint8), 
			target_names=["others", class_name])

		print_and_log(report)

		## oof preds
		# oof_preds[valid_index] = p_valid
		

		del conv_trn_feat,conv_valid_feat, X_valid
		
		
		
		
		min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
		print_and_log( 'Minimum loss at epoch'+ '{:d}'.format(idx+1)+ '='+ '{:.4f}'.format(min_val_loss))

		max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
		print_and_log('Maximum accuracy at epoch'+ '{:d}'.format(idx+1) + '='+ '{:.4f}'.format(max_val_acc))

		val_errors.append(min_val_loss)
		val_accuracies.append(max_val_acc)
		

		############################# TEST PREDICTION
		# del X_train, Y_train, X_valid, Y_valid

		## doing test predictions 
		print "Predicting test files"
		
		p_test = make_staged_predictions(model,num_parts=5,do_augmentation=False)        
		yfull_test.append(p_test)

		del model # make sure we don't resue weights 
		
		

	print_and_log( "Done kfolding for class {}".format(class_name)) 
	save_array("data/cache/{}/test_preds_5cv_vgg1.dat".format(class_name),yfull_test)	
	# np.save("data/cache/oof_vgg1.dat",oof_preds)


	thresholds = np.array(thresholds,np.float16)
	save_array("data/cache/{}/thresholds_vgg1.dat".format(class_name),thresholds)


	print_and_log ("best val_loss has a mean of {} and stdev of {}".format(np.mean(val_errors),np.std(val_errors)))
	print_and_log ("best val_acc has a mean of {} and stdev of {}".format(np.mean(val_accuracies) , np.std(val_accuracies)))
	print_and_log ("local best thresholds have a mean of {} and stdev of {}".format(thresholds.mean(),  thresholds.std()))

	del logger