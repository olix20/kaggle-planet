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


class MixIterator(Iterator):
    
    def __init__(self, iters):
        self.iters = iters
        self.multi = type(iters) is list
        self.N = len(x_train)
#         if self.multi:
#             self.N = sum([it[0].N for it in self.iters])
#         else:
#             self.N = sum([it.N for it in self.iters])
        super(MixIterator, self).__init__(x_train.shape[0], 128, True, seed=None)

    def reset(self):
        for it in self.iters: it.reset()

    def __iter__(self):
        return self

    def next(self, *args, **kwargs):
#         if self.multi:
#             nexts = [[next(it) for it in o] for o in self.iters]
#             n0s = np.concatenate([n[0] for n in o])
#             n1s = np.concatenate([n[1] for n in o])
#             return (n0, n1)
#         else:
        nexts = [next(it) for it in self.iters]
        n0 = np.concatenate([n[0] for n in nexts])
        n1 = np.concatenate([n[1] for n in nexts])
        return (n0, n1)




#### Load data
x_train = load_array("data/cache/xtrain_{}x{}.dat".format(target_size[0],target_size[1]))
y_train = load_array("data/cache/ytrain.dat")

VALIDATION_SPLIT = 0.2
np.random.seed(3)





perm = np.random.permutation(len(x_train))
idx_train = perm[:int(len(x_train)*(1-VALIDATION_SPLIT))]
idx_val = perm[int(len(x_train)*(1-VALIDATION_SPLIT)):]


# X_train = x_train[idx_train]
# Y_train = y_train[idx_train]


X_valid = x_train[idx_val]
Y_valid = y_train[idx_val]

# del x_train


x_test = np.memmap("data/cache/xtest_{}x{}.memmapped".format(target_size[0],target_size[0]), dtype='float32', mode='r', 
                       shape=(df_test.shape[0],target_size[0],target_size[0],3))
y_test = load_array('data/cache/preds_resnet4a_12xtta_bugfixed.dat/')




#### Model



resnet = ResNet50(include_top=False, input_shape=(target_size[0], target_size[0],3))


for layer in resnet.layers:
    layer.trainable = False

for i in range(-95,0):
    # 4a: -95
    # tuning 5b: -21
    # tuning branch 5c -11
    # tuning from 5a: -33
    # 3a: -137
    # 2a: -169
    resnet.layers[i].trainable = True





x = Flatten()(resnet.layers[-1].output)
x = Dropout(0.5)(x)

x = Dense(1024, activation='relu')(x)

x = Dropout(0.5)(x)

x = Dense(17,activation='sigmoid',name="fc17")(x)






model = Model(inputs=resnet.input,outputs=x)
model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights("weights/resnet_augv2_17class_4a_pseudo.h5")






batch_size = 128

train_gen = ImageDataGenerator2(rotation_range=270,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
    shear_range=0.1,
#     zoom_range=0.1,
    channel_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)




train_batch = train_gen.flow(x_train,y_train,batch_size=96,shuffle=True)
# valid_batch = train_gen.flow(X_valid,Y_valid,batch_size=20,shuffle=True)
test_batch = train_gen.flow(x_test,y_test,batch_size=32,shuffle=True)
mi = MixIterator([train_batch,test_batch])





kfold_weights_path = os.path.join('weights/', 'resnet_augv2_17class_4a_pseudo_plus_validset_2epochs.h5')


callbacks = [EarlyStopping(monitor='val_loss', patience=6),
             ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True),
             ReduceLROnPlateau(monitor='val_loss',  patience=2, verbose=1, factor=0.1, min_lr=1e-6)
            ]




history = model.fit_generator(mi, validation_data=(X_valid, Y_valid),
                               steps_per_epoch=len(x_train)/batch_size,epochs=2,callbacks=callbacks)


min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print ('Minimum loss at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(min_val_loss))

max_val_acc, idx = max((val, idx) for (idx, val) in enumerate(history.history['val_acc']))
print ('Maximum accuracy at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(max_val_acc))
