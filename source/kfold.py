import utils
from utils import *

from tqdm import tqdm 
import psutil
from sys import getsizeof


nf=128; p=0.4
target_size = (100,100)#(256, 256)
image_feature_size = (256, 25, 25)


n_folds = 5
num_augmentation_sets_per_fold = 2

current_fold = 0
sum_score = 0

yfull_test = []
# yfull_train =[]
thresholds = []
oof_preds = np.zeros((x_train.shape[0],17))


val_errors = []
val_accuracies = []





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
	return [
		BatchNormalization(axis=1, input_shape=(256, 25, 25)),#conv_layers[-1].output_shape[1:]),
		Convolution2D(nf,(3,3), activation='relu', padding='valid'),
		Dropout(p/2),

#         MaxPooling2D(),

		BatchNormalization(axis=1),

#         MaxPooling2D(),
		Convolution2D(nf,(3,3), activation='relu', padding='valid'),
		Dropout(p/2),

		BatchNormalization(axis=1),
#         MaxPooling2D(),
		Convolution2D(nf,(3,3), activation='relu', padding='valid'),
		Dropout(p/2),

		BatchNormalization(axis=1),

		
#         MaxPooling2D(),
		Convolution2D(17,(3,3), padding='same'),
		Dropout(p),
#         GlobalAveragePooling2D(),
		GlobalMaxPooling2D(),
		Activation('softmax')
	]


def create_training_pretrained_features():
	current_fold = 0

	x_train = load_array("data/cache/xtrain_100x100.dat") # we wouldn't need to load this any more 


	vgg_common = create_base_vgg()
	vgg_common.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])

	train_gen = image.ImageDataGenerator( 
		rotation_range=0.1,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.05,
		zoom_range=0.05,
		channel_shift_range=0.05,
		horizontal_flip=True,
		vertical_flip=True)


	kf = KFold( n_splits=n_folds, shuffle=True, random_state=2020)


	for train_index, valid_index in kf.split(x_train):
			
		X_train = x_train[train_index]
		data_length = len(X_train)

		current_fold += 1

		train_data = np.memmap("data/cache/vgg_features_train_100x100_part{}of{}.planet".\
			format(current_fold,n_folds)
			, dtype='float32', mode='w+', 
			shape=(num_augmentation_sets_per_fold*data_length,)+ image_feature_size)



		print ("creating augmented features for round ", i)
		train_batch = train_gen.flow(X_train,Y_train,batch_size=32,shuffle=False,seed=current_fold)



		for j in range(1,num_augmentation_sets_per_fold+1):
			start_index = (j-1)*data_len
	        end_index = start_index + data_len

			print ("augmentation round ", j)

			train_data[start_index:end_index,]   = vgg_common.predict_generator(train_batch, 
				steps=math.ceil(data_length/32.0) ,verbose=1) 
		


		# save data to disk	
		train_data.flush()
		train_data.close()	



def create_test_pretrained_features(num_parts=5,num_augmentations=10):
	current_fold = 0

	data = load_array("data/cache/xtest_100x100.dat")


	test_data_memp = np.memmap("data/cache/vgg_features_test_100x100_allinone_{}sets.planet".format(n_folds)
		, dtype='float32', mode='w+', 
		shape=(num_augmentations*len(data,)+ image_feature_size)


	vgg_common = create_base_vgg()
	vgg_common.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])

	test_gen = image.ImageDataGenerator( 
		rotation_range=0.1,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.05,
		zoom_range=0.05,
		channel_shift_range=0.05,
		horizontal_flip=True,
		vertical_flip=True)


	chunck_size = len(data)/num_parts

    print "data has shape: ", data.shape
    print "breaking data into chunks of ", chunck_size




	for n in range(num_augmentations): # 2 sets per training folds (5 folds > 2 test predictions per each fold)
            print ""
            print "augmentation round {} of {}".format(n+1,num_augmentations) 
            partial_preds = []
            
            test_batch = test_gen.flow(data[start_index:end_index],batch_size=16,shuffle=False, seed=n)


            for i in range(1,num_parts+2):
                print ""
                print "predicting part {} of {}".format(i,num_parts)      
                
                start_index = n*len(x_test) + (i-1)*chunck_size
                end_index = n*len(x_test) + min(len(data),i*chunck_size)
                num_items = end_index-start_index


                test_data_memp[start_index:end_index,] = vgg_common.predict_generator(test_batch, 
                	steps=math.ceil(num_items/16.0) ,verbose=1) 


            test_data_memp.flush()
            test_data_memp.close()




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



	x_train = load_array("data/cache/xtrain_100x100.dat")
	y_train = load_array("data/cache/ytrain.dat")





	################################ KFOLD ###########################3


		   
	kf = KFold( n_splits=n_folds, shuffle=True, random_state=2020)

	for train_index, valid_index in kf.split(x_train,y_train):
			# start_time_model_fitting = time.time()
			
		X_train = x_train[train_index]
		Y_train = y_train[train_index]
		X_valid = x_train[valid_index]
		Y_valid = y_train[valid_index]
		
		data_length = len(X_train)


		print ("training X and y: ", X_train.shape,Y_train.shape)
		print ("training X and y: ", X_valid.shape,Y_valid.shape)

		current_fold += 1
		print('Start KFold number {} from {}'.format(current_fold, nfolds))
		print('Split train: ', len(X_train), len(Y_train))
		print('Split valid: ', len(X_valid), len(Y_valid))
		
		kfold_weights_path = os.path.join('weights/', 'vgg1_weights_kfold_' + str(current_fold) + '.h5')
		callbacks = [EarlyStopping(monitor='val_loss', patience=4),
					 ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True)]
 




		##################   TRAINING

		vgg_common = create_base_vgg()
		vgg_common.compile(Adam(), 'categorical_crossentropy', metrics=['accuracy'])

		conv_trn_feat = np.memmap("data/cache/vgg_features_train_100x100_part{}of{}.planet".\
			format(current_fold,n_folds)
			, dtype='float32', mode='r', 
			shape=(num_augmentation_sets_per_fold*data_length,)+ image_feature_size)


		Y_train = np.concatenate([Y_train]*num_augmentation_sets_per_fold) #duplicate targets too
		


		print "create conv_valid_feat"		
		conv_valid_feat = vgg_common.predict(X_valid, batch_size=32, verbose=1)



		print "Training round {}".format(current_fold)
		model =  Sequential(get_lrg_layers())
		model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

		
		
		######################## ACTION
		history = model.fit(x = conv_trn_feat, y= Y_train, validation_data=(conv_valid_feat, Y_valid),
		  batch_size=32, epochs=1,callbacks=callbacks,
		  shuffle=True,verbose=1)

		
		
		######################## VALIDATION 
		
		if os.path.isfile(kfold_weights_path):
			print ""
			print ("loading best weights from ",kfold_weights_path)
			model.load_weights(kfold_weights_path)
			
		
		## verifying local validation results
		p_valid = model.predict(conv_valid_feat, batch_size = 32, verbose=1)
		# print("local score with threshold 0.08: ",f2_score(Y_valid, np.array(p_valid) > 0.08))
		# print("Optimizing prediction threshold")
		local_best_thresholds = optimise_f2_thresholds(Y_valid, p_valid)
#         print(local_best_thresholds)
		thresholds.append(local_best_thresholds)

		
		## oof preds
		oof_preds[valid_index] = p_valid
		del conv_trn_feat,conv_valid_feat 
		
		
		
		
		min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
		print 'Minimum loss at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(min_val_loss)

		max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
		print 'Maximum accuracy at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(max_val_acc)
		
		val_errors.append(min_val_loss)
		val_accuracies.append(max_val_acc)
		

		############################# TEST PREDICTION
		del X_train, Y_train, X_valid, Y_valid

		## doing test predictions 
		print "Predicting test files"
		
		p_test = make_staged_predictions(x_test,num_parts=5,do_augmentation=True)        
		yfull_test.append(p_test)

		del model # make sure we don't resue weights 
		
		break
		

	print "Done kfolding!"        
	np.save("data/cache/test_preds_10cv_vgg1_.dat",yfull_test)	
	np.save("data/cache/oof_vgg1.dat",oof_preds)


	thresholds = np.array(thresholds,np.float16)
	np.save("data/cache/thresholds_vgg1.dat",thresholds)


	print ("best val_loss has a mean of {} and stdev of {}".format(np.mean(val_errors), np.std(val_errors)))
	print ("best val_acc has a mean of {} and stdev of {}".format(np.mean(val_accuracies), np.std(val_accuracies)))
	print ("local best thresholds have a mean of {} and stdev of {}".format(thresholds.mean(), thresholds.std()))

