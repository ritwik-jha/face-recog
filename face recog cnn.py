#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.applications import vgg16


# In[2]:


model = vgg16.VGG16(weights='imagenet'  , include_top=False, input_shape=(224,224,3))


# In[3]:


model.summary()


# In[4]:


for l in model.layers:
    l.trainable=False


# In[5]:


x= model.output


# In[6]:


from keras.layers import Dense , Flatten


# In[7]:


x = Flatten()(x)


# In[8]:


x = Dense(units = 1024 , activation='relu')(x)


# In[9]:


x = Dense(units = 512 , activation='relu')(x)


# In[10]:


x = Dense(units = 1 , activation = 'sigmoid')(x)


# In[11]:


from keras.models import Model


# In[12]:


nmodel = Model(inputs=model.input , outputs=x)


# In[13]:


nmodel.summary()


# In[14]:


from keras.optimizers import adam


# In[15]:


nmodel.compile(optimizer=adam() , loss='binary_crossentropy' , metrics=['accuracy'])


# In[16]:


from keras_preprocessing.image import ImageDataGenerator


# In[17]:


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        'dataset_facerecog/train/',
        target_size=(224,224),
        batch_size=32,
        class_mode='binary')
test_set = test_datagen.flow_from_directory(
        'dataset_facerecog/test/',
        target_size=(224,224),
        batch_size=32,
        class_mode='binary')
nmodel.fit(
        training_set,
        steps_per_epoch=8,
        epochs=2,
        validation_data=test_set,
        validation_steps=800)


# In[1]:


from keras.preprocessing import image


# In[3]:


im = image.load_img('test.png' , target_size=(224,224))


# In[4]:


im = image.img_to_array(im)


# In[5]:


im.shape()


# In[6]:


import numpy as np


# In[7]:


fimg = np.expand_dims(im , axis=0)


# In[8]:


result = nmodel.predict(fimg)


# In[ ]:


if relult == 1:
    print('ritwik')
else:
    print('raunak')

