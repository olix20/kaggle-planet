import utils
from utils import *
from keras.applications.resnet50 import *
from keras.preprocessing import image
from keras.preprocessing.image import *

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from image_90rotations import *
# from ImageDataGenerator_extended2 import *
target_size = (200,200)#(256, 256)




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





resnet = ResNet50(include_top=False, input_shape=(target_size[0], target_size[0],3))
for layer in resnet.layers:
    layer.trainable = False

x = Flatten()(resnet.layers[-1].output)
x = Dropout(0.5)(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.5)(x)

x = Dense(17,activation='sigmoid',name="fc17")(x)

model = Model(inputs=resnet.input,outputs=x)
model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("weights/resnet_augv2_17class_4a.h5")



batch_size = 128

train_gen = ImageDataGenerator2(rotation_range=270,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
    shear_range=0.1,
#     zoom_range=0.1,
    channel_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)



def read_partial_xtest(start,end):
    c = 0
    num_items = end-start
    x_test = np.zeros((num_items,target_size[0],target_size[0],3)).astype(np.float32)

    for f, tags in tqdm(df_test.iloc[start:end].values, miniters=1000):

        img = image.load_img('data/test-jpg/{}.jpg'.format(f), target_size=target_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        x_test[c] = x
        c +=1

    return x_test





def make_staged_predictions_v2(model,num_parts, nb_aug=1):
    partial_preds = []

    batch_size = 128.0
    data_length = df_test.shape[0]#X_valid[0:16].shape[0]
    chunck_size = data_length//num_parts

    print( "breaking data into chunks of ", chunck_size)
    
#     ipdb.set_trace()
    final_preds = np.zeros((data_length,17)).astype(np.float32)
#     data = X_valid[0:16]


    for i in tqdm(range(1,num_parts+2)):
        print( "predicting part {} of {}".format(i,num_parts)      )

        start_index = (i-1)*chunck_size
        end_index = min(data_length,i*chunck_size)
        num_items = end_index-start_index
        data = read_partial_xtest(start_index,end_index)
        partials = []
        
        
        for n in tqdm(range(nb_aug)):
            print ("augmentation round {} of {}".format(n+1,nb_aug) )
#             partial_preds = np.zeros((num_items,17)).astype(np.float32)

            test_batch = train_gen.flow(data,batch_size=int(batch_size),shuffle=False)
            t = model.predict_generator(test_batch,steps=math.ceil(num_items/batch_size), verbose=1)[:num_items]

            partials.append(t)
#             partial_preds += t 
#         partial_preds /= float(nb_aug) 
            
        final_preds[start_index:end_index] = np.mean(partials,axis=0)
        save_array("data/cache/resnet_ft4a_final_preds_temp_part{}".format(i),final_preds)   
    
    return final_preds




preds_x12 = make_staged_predictions_v2(model,num_parts=5,nb_aug=12)

save_array("data/cache/preds_resnet4a_12xtta_bugfixed.dat", preds_x12)


print ("done!")