import numpy as np
import random
import json
from glob import glob
from keras.models import model_from_json,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import  ModelCheckpoint,Callback,LearningRateScheduler
import keras.backend as K
from losses import *
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
from keras.callbacks import History 
from keras.optimizers import SGD
from keras.layers import Dropout,GaussianNoise, Input,Activation
from numpy import random

from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras import initializers


import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)


class SGDLearningRateTracker(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.get_value(optimizer.lr)
        decay = K.get_value(optimizer.decay)
        lr=lr/10
        decay=decay*10
        K.set_value(optimizer.lr, lr)
        K.set_value(optimizer.decay, decay)
        print('LR changed to:',lr)
        print('Decay changed to:',decay)



class Training(object):
    
    history = History()

    def __init__(self, batch_size,nb_epoch,load_model_resume_training):

        self.batch_size = batch_size
        self.nb_epoch = nb_epoch

        #loading model from path to resume previous training without recompiling the whole model
        if load_model_resume_training is not None:
            unet = tf.keras.applications.InceptionResNetV2(include_top=False,weights='imagenet',input_shape=(150,150,3),classifier_activation="softmax")
            #unet = Unet_model(img_shape=(240, 240, 4), load_model_weights=load_model_resume_training)
            self.model = unet               
        
            #self.model =load_model(load_model_resume_training,custom_objects={'gen_dice_loss': gen_dice_loss,'dice_whole_metric':dice_whole_metric,
                                                                              #'dice_core_metric':dice_core_metric,'dice_en_metric':dice_en_metric})
            print("pre-trained model loaded!")
        else:

            inputs = Input((128,128,4))
            conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(inputs)
            conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(pool1)
            conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

            conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(pool2)
            conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(pool3)
            conv4 = Conv2D(256, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

            conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(pool4)
            conv5 = Conv2D(512, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(conv5)

            up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same',
                        kernel_initializer=initializers.random_normal(stddev=0.01))(conv5),conv4], axis=3)
            conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(up6)
            conv6 = Conv2D(256, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(conv6)

            up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same',
                        kernel_initializer=initializers.random_normal(stddev=0.01))(conv6),conv3], axis=3)
            conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(up7)
            conv7 = Conv2D(128, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(conv7)

            up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2),padding='same',
                        kernel_initializer=initializers.random_normal(stddev=0.01))(conv7),conv2], axis=3)
            conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(up8)
            conv8 = Conv2D(64, (3, 3), activation='relu', padding='same',)(conv8)

            up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same',
                        kernel_initializer=initializers.random_normal(stddev=0.01))(conv8),conv1], axis=3)
            conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(up9)
            conv9 = Conv2D(32, (3, 3), activation='relu', padding='same',
                          kernel_initializer=initializers.random_normal(stddev=0.01))(conv9)

            conv10 = Conv2D(4, (1, 1), activation='relu',
                            kernel_initializer=initializers.random_normal(stddev=0.01))(conv9)
            conv10 = Activation('softmax')(conv10)
            self.model = Model(inputs=[inputs], outputs=[conv10])

            
            print("U-net CNN compiled!")

                    
    def fit_unet(self,X33_train,Y_train,X_patches_valid,Y_labels_valid):

        train_generator=self.img_msk_gen(X33_train,Y_train,9999)
        checkpointer = ModelCheckpoint(filepath='./ResUnet.{epoch:02d}_{val_loss:.4f}_{val_accuracy:.4f}.hdf5', verbose=1)
        # Compile model
        sgd = SGD(lr=0.08, momentum=0.9, decay=5e-6, nesterov=False)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy',f1_m,precision_m, recall_m])
        
        #history = self.model.fit_generator(train_generator,steps_per_epoch=len(X33_train)//self.batch_size,epochs=self.nb_epoch, validation_data=(X_patches_valid,Y_labels_valid),verbose=1, callbacks = [checkpointer,SGDLearningRateTracker()])
        print("From history")
        print(X33_train[0].shape)
        print(Y_train[0].shape)
        print(X_patches_valid.shape)
        print(Y_labels_valid.shape)
        history = self.model.fit(X33_train,Y_train, epochs=self.nb_epoch,batch_size=self.batch_size,validation_data=(X_patches_valid,Y_labels_valid),verbose=1, callbacks = [checkpointer,SGDLearningRateTracker()])
        print("Evaluate")
        result = self.model.evaluate(X_patches_valid, Y_labels_valid, batch_size=self.batch_size)
        dict(zip(self.model.metrics_names, result))
        
        
        # list all data in history
        print(history.history.keys())
        
        print("\naccuracy : " , history.history['accuracy'])
        print("val_accuracy : " ,history.history['val_accuracy'])
        
        
        print("\nloss : ",history.history['loss'])
        print("val_loss : ",history.history['val_loss'])
        

        print("\nf_score : " , history.history['f1_m'])
        print("val_f_score : " ,history.history['val_f1_m'])
        
        print("\nprecision_m : " , history.history['precision_m'])
        print("val_precision_m : " ,history.history['val_precision_m'])
        
        
        print("\nrecall_m : " , history.history['recall_m'])
        print("val_recall_m : " ,history.history['val_recall_m'])
        
        # summarize history for accuracy
        fig = plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        fig.savefig('/content/drive/MyDrive/Transfer Based Model/accuracy_figure.png')
        
        # summarize history for loss
        fig1 = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        fig1.savefig('/content/drive/MyDrive/Transfer Based Model/loss_figure.png')
        
        # summarize history for nf_score
        fig1 = plt.figure()
        plt.plot(history.history['f1_m'])
        plt.plot(history.history['val_f1_m'])
        plt.title('model f_score')
        plt.ylabel('f_score')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        fig1.savefig('/content/drive/MyDrive/Transfer Based Model/f_score_figure.png')
        
        # summarize history for precision
        fig1 = plt.figure()
        plt.plot(history.history['precision_m'])
        plt.plot(history.history['val_precision_m'])
        plt.title('model precision')
        plt.ylabel('precision')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        fig1.savefig('/content/drive/MyDrive/Transfer Based Model/precision_figure.png')
        
        # summarize history for recall
        fig1 = plt.figure()
        plt.plot(history.history['recall_m'])
        plt.plot(history.history['val_recall_m'])
        plt.title('model recall')
        plt.ylabel('recall_m')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        fig1.savefig('/content/drive/MyDrive/Transfer Based Model/recall_figure.png')
        
        
    def img_msk_gen(self,X33_train,Y_train,seed):

        '''
        a custom generator that performs data augmentation on both patches and their corresponding targets (masks)
        '''
        datagen = ImageDataGenerator(horizontal_flip=True,data_format="channels_last")
        datagen_msk = ImageDataGenerator(horizontal_flip=True,data_format="channels_last")
        image_generator = datagen.flow(X33_train,batch_size=4,seed=seed)
        y_generator = datagen_msk.flow(Y_train,batch_size=4,seed=seed)
        while True:
            yield(image_generator.next(), y_generator.next())


    def save_model(self, model_name):
        '''
        INPUT string 'model_name': path where to save model and weights, without extension
        Saves current model as json and weights as h5df file
        '''

        model_tosave = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        json_string = self.model.to_json()
        self.model.save_weights(weights)
        with open(model_tosave, 'w') as f:
            json.dump(json_string, f)
        print ('Model saved.')

    def load_model(self, model_name):
        '''
        Load a model
        INPUT  (1) string 'model_name': filepath to model and weights, not including extension
        OUTPUT: Model with loaded weights. can fit on model using loaded_model=True in fit_model method
        '''
        print ('Loading model {}'.format(model_name))
        model_toload = '{}.json'.format(model_name)
        weights = '{}.hdf5'.format(model_name)
        with open(model_toload) as f:
            m = next(f)
        model_comp = model_from_json(json.loads(m))
        model_comp.load_weights(weights)
        print ('Model loaded.')
        self.model = model_comp
        return model_comp



if __name__ == "__main__":
    #set arguments

    #reload already trained model to resume training
    #model_to_load="./ResUnet.03_0.677.hdf5"
    model_to_load=None
    #save=None
    #compile the model
    brain_seg = Training(batch_size=8,nb_epoch=5,load_model_resume_training=model_to_load)

    print("number of trainabale parameters:",brain_seg.model.count_params())
    print(brain_seg.model.summary())
    plot_model(brain_seg.model, to_file='./transfer_Model_Architecture.png', show_shapes=True)
    #print("Done Ploting")
    #load data from disk
    Y_labels=np.load("/content/drive/MyDrive/Transfer Based Model/y.npy").astype(np.uint8)
    X_patches=np.load("/content/drive/MyDrive/Transfer Based Model/x.npy").astype(np.float32)
    Y_labels_valid=np.load("/content/drive/MyDrive/Transfer Based Model/vy.npy").astype(np.uint8)
    X_patches_valid=np.load("/content/drive/MyDrive/Transfer Based Model/vx.npy").astype(np.float32)
    print(X_patches.shape)
    print(len(Y_labels))
    
    #Y_L1 = random.choice([0,1], size=(len(Y_labels)))
    #Y_L2 = random.choice([0,1], size=(len(Y_labels_valid)))
    #print(Y_L2.shape)
    # fit model
    brain_seg.fit_unet(X_patches,Y_labels , X_patches_valid , Y_labels_valid)#*
    
    brain_seg.save_model('./Xception_Training_Model')
    #print("Model Saved")




