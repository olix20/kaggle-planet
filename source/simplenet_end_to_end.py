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
import keras
from keras.layers.merge import *



def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
	num_classes = 17
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


channel_means = np.array([80.301086, 87.727798, 77.089554], dtype=np.float32).reshape((3,1,1))

def preprocess(x):
    return x - channel_means
#     return x/255.0

#3 minutes per epoch, 0.914 F2 score, 0.1094 val loss
def make_conv_bn_relu(x,num_layers,num_filters,kernel_size):
    
    for _ in range(num_layers):
    
        x = Convolution2D(num_filters,kernel_size,padding='same')(x)
        x = BatchNormalization(axis=1)(x)
        x = Activation(activation="relu")(x)
    
    return x

def make_bottleneck(x,num_filters,double_output_filters=True):
    
    x = Convolution2D(num_filters,1,padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation(activation="relu")(x)

    
    x = Convolution2D(num_filters,3,padding='same')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation(activation="relu")(x)
    
    
    if double_output_filters:	
        x = Convolution2D(num_filters*2,1,padding='same')(x)
    else:
        x = Convolution2D(num_filters,1,padding='same')(x)

    x = BatchNormalization(axis=1)(x)
    x = Activation(activation="relu")(x)
    

    return x 


def simple_net():
    model_input = Input(shape=(3,target_size[0],target_size[1]))
    
    #preprocess
    x = Lambda(preprocess,output_shape=(3,)+target_size)(model_input)
    x = make_conv_bn_relu(x,3,16,(1,1))
    
    
    
    x = make_bottleneck(x,32)
    max1  = MaxPooling2D(padding='same')(x)
    
    x = make_bottleneck(max1,64)
    max2  = MaxPooling2D(padding='same')(x)
    
    x = make_bottleneck(max2,128,False)
    max3  = MaxPooling2D(padding='same')(x)

#     x = make_bottleneck(max3,256,False)
#     max4  = MaxPooling2D(padding='same')(x)

        
    all_avg_features = concatenate([GlobalAveragePooling2D()(max1),
                GlobalAveragePooling2D()(max2),
                GlobalAveragePooling2D()(max3)])
    
#     all_max_features = concatenate([GlobalMaxPooling2D()(max1),
#             GlobalMaxPooling2D()(max2),
#             GlobalMaxPooling2D()(max3)])
    

        
    #,GlobalAveragePooling2D()(max4)
    
    x = Dense(512,activation='relu')(all_avg_features)
    x = Dropout(0.5)(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(17,activation='sigmoid')(x)
    
    
    return Model(inputs=model_input, outputs=x)
   


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



def make_staged_predictions(model,num_parts, nb_aug=1):
    partial_preds = []

    data = load_array("data/cache/xtest_{}x{}.dat".format(target_size[0],target_size[0]))

    chunck_size = len(data)/num_parts

    print "data has shape: ", data.shape
    print "breaking data into chunks of ", chunck_size
    
    
    augmented_predictions = np.zeros((data.shape[0],17)).astype(np.float16)

    for n in range(nb_aug):
        print "augmentation round {} of {}".format(n+1,nb_aug) 
        partial_preds = []
        batch_size = 128.0

        for i in range(1,num_parts+2):
            print "predicting part {} of {}".format(i,num_parts)      

            start_index = (i-1)*chunck_size
            end_index = min(len(data),i*chunck_size)
            num_items = end_index-start_index

            test_batch = train_gen.flow(data[start_index:end_index],batch_size=int(batch_size),shuffle=False)
            pp = model.predict_generator(test_batch,steps=math.ceil(num_items/batch_size), verbose=0)

            print "items needed vs. items generated: ",num_items,len(pp) 
            
            partial_preds.append(pp)


        augmented_predictions += np.vstack(partial_preds)

    augmented_predictions /= (nb_aug) 
    return augmented_predictions




if __name__ == "__main__":
	global target_size
	global image_feature_size


	logger = None

	target_size = (56,56)#(256, 256)


	############# prep
	# global logger




	VALIDATION_SPLIT = 0.1
	np.random.seed(2089)



	# create_logger("main")
	current_fold = 0 
	n_folds = 5
	num_augmentation_sets_per_fold = 3




	yfull_test = []
	# yfull_train =[]
	thresholds = []
	# oof_preds = np.zeros((x_train.shape[0],17))


	val_errors = []
	val_accuracies = []



	############# Action

	# class_index = [label_map[class_name]]


	K.set_image_data_format('channels_first')
	K.image_data_format()



	df_train = pd.read_csv('data/train_v2.csv')
	df_test = pd.read_csv('data/sample_submission_v2.csv')

	flatten = lambda l: [item for sublist in l for item in sublist]
	labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

	

	## splitting data
	np.random.seed(2089)

	x_train = load_array("data/cache/xtrain_{}x{}.dat".format(target_size[0],target_size[1]))
	y_train = load_array("data/cache/ytrain.dat")


	perm = np.random.permutation(len(x_train))
	idx_train = perm[:int(len(x_train)*(1-VALIDATION_SPLIT))]
	idx_val = perm[int(len(x_train)*(1-VALIDATION_SPLIT)):]

	X_train = x_train[idx_train]
	X_valid = x_train[idx_val]


	Y_train = y_train[idx_train]
	Y_valid = y_train[idx_val]


	train_gen = image.ImageDataGenerator( 
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True)






	### TRAINING 

	kfold_weights_path = os.path.join('weights/', 'simplenet_BN_relu_56_trainx4.h5')


	#              ReduceLROnPlateau(monitor='val_loss',  patience=3, verbose=1, factor=0.1, min_lr=1e-7)
	callbacks = [EarlyStopping(monitor='val_loss', patience=3),
	             ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True)
	            ]
	model =  simple_net()
	model.compile(optimizer="nadam", loss='binary_crossentropy', metrics=['accuracy'])

	batch_size = 256






	# history = model.fit_generator(train_gen.flow(X_train, Y_train,batch_size=batch_size,shuffle=True), 
	# 	validation_data=(X_valid, Y_valid),
	# 	steps_per_epoch=num_augmentation_sets_per_fold*len(X_train)/batch_size,
	# 	epochs=50,
	# 	callbacks=callbacks)




	# ### Results and calculating thresholds 

	# min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
	# print 'Minimum loss at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(min_val_loss)

	# max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
	# print 'Maximum accuracy at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(max_val_acc)



	if os.path.isfile(kfold_weights_path):
	    print "loading best weights from '{}'".format(kfold_weights_path)
	    model.load_weights(kfold_weights_path)
	    
	preds_val = model.predict(X_valid)

	thres = optimise_f2_thresholds(Y_valid, preds_val)
	print('F2 Score:', f2_score(Y_valid, preds_val>thres))
	save_array("data/cache/simplenet_thres.dat",np.array(thres).astype(np.float16))



	############## Making predictions 

	# del x_train, X_train, X_valid #, conv_trn_feat, conv_valid_feat

	# preds = make_staged_predictions(model,num_parts=5,nb_aug=5)
	# save_array("data/cache/preds_simplenet_2xaug_5xtaug.dat",preds)




	print "done!"


	# del logger