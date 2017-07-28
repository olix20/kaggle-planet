import utils
from utils import *

from tqdm import tqdm 

target_size = (100,100)#(256, 256)




def make_staged_predictions_forend2end(model, data,num_parts, do_augmentation=True):
    partial_preds = []
    nb_aug = 1

    chunck_size = len(data)/num_parts

    print "data has shape: ", data.shape
    print "breaking data into chunks of ", chunck_size
    

    normal_image_preds = np.zeros((data.shape[0],17))
    
    
    if not do_augmentation:
        return   None  

    
    
    else: ## test augmentation 
        augmented_predictions = normal_image_preds 
        for n in range(nb_aug):
            print ""
            print "augmentation round {} of {}".format(n+1,nb_aug) 
            partial_preds = []

            for i in range(1,num_parts+2):
                print ""
                print "predicting part {} of {}".format(i,num_parts)      
                
                start_index = (i-1)*chunck_size
                end_index = min(len(data),i*chunck_size)
                num_items = end_index-start_index
                
                
                test_batch = train_gen.flow(data[start_index:end_index],batch_size=32,shuffle=False)
                
                vgg_test_feat = model.predict_generator(test_batch, steps=math.ceil(num_items/32.0) ,verbose=1) 
                
#                 print ""
#                 print ("items needed vs. items generated: ",num_items,len(vgg_test_feat) )
                
                vgg_test_feat = vgg_test_feat[:num_items] #to make sure we only take as much as we need for this batch 
                
                partial_preds.append(vgg_test_feat)
                
                
            augmented_predictions += np.vstack(partial_preds)
            
        augmented_predictions /= (nb_aug)
        return augmented_predictions

    
    


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


if __name__ == "__main__":


	K.set_image_data_format('channels_first')
	K.image_data_format()

	x_train = []
	y_train = []

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


	train_gen = image.ImageDataGenerator( 
		rotation_range=0.1,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.1,
		zoom_range=0.1,
		channel_shift_range=0.2,
		horizontal_flip=True,
		vertical_flip=True)

	x_train = load_array("data/cache/xtrain_100x100.dat")
	y_train = load_array("data/cache/ytrain.dat")


	nf=128; p=0.4







	################################ KFOLD ###########################3
	nfolds = 10
	batch_size = 32
	current_fold = 0
	sum_score = 0

	yfull_test = []
	# yfull_train =[]
	thresholds = []
	oof_preds = np.zeros((x_train.shape[0],17))


	val_errors = []
	val_accuracies = []

		   
	kf = KFold( n_splits=nfolds, shuffle=True, random_state=2020)

	for train_index, valid_index in kf.split(x_train,y_train):
			# start_time_model_fitting = time.time()
			
		# X_train = x_train[train_index]
		# Y_train = y_train[train_index]
		X_valid = x_train[valid_index]
		Y_valid = y_train[valid_index]


		current_fold += 1

		if current_fold > 4:
			continue

		end_to_end_model = create_base_vgg(True)


		print('Start pred round  {} from {}'.format(current_fold, nfolds))
		
		kfold_weights_path = os.path.join('weights/', 'vgg1_weights_kfold_' + str(current_fold) + '.h5')
		callbacks = [EarlyStopping(monitor='val_loss', patience=4),
					 ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True)]
 
		##################   TRAINING

		end_to_end_model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])

		
		end_to_end_model.load_weights(kfold_weights_path)
			

		## doing test predictions 
		print("Evaluating performance on validation set ")
		

		## verifying local validation results
		p_valid = end_to_end_model.predict(X_valid, batch_size = 64, verbose=1)
		print("local score with threshold 0.08: ",f2_score(Y_valid, np.array(p_valid) > 0.08))
		print("Optimizing prediction threshold")
		local_best_thresholds = optimise_f2_thresholds(Y_valid, p_valid)
#         print(local_best_thresholds)
		thresholds.append(local_best_thresholds)



		############################# TEST PREDICTION
		## doing test predictions 
		print("Predicting test files")

		del  X_valid, Y_valid
		x_test = load_array("data/cache/xtest_100x100.dat")


		
		p_test = make_staged_predictions_forend2end(end_to_end_model,x_test,num_parts=5,do_augmentation=True)        
		yfull_test.append(p_test)

		del end_to_end_model # make sure we don't resue weights 
		
		

	print "Done kfolding!"        
	np.save("data/cache/test_preds_10cv_vgg1_.dat",yfull_test)	


	thresholds = np.array(thresholds,np.float16)
	np.save("data/cache/thresholds_vgg1.dat",thresholds)
	

	print ("local best thresholds have a mean of {} and stdev of {}".format(thresholds.mean(axis=0), thresholds.std(axis=0)))

