## Getting it started ##
import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf
import re
from PIL import Image
# from sklearn.model_selection import train_test_split
from keras.preprocessing import image

Image_length=300
Image_width=300
Image_dimension=(Image_width,Image_length)

##Image Wrangling##
img_path='./Saturday One on One/garbage_classification/'
our_classes={0:'battery',1:'biological',2:'brown-glass',3:'white-glass',4:'cardboard',5:'clothes',
             6:'green-glass',7:'metal',8:'paper',9:'plastic',10:'shoes',11:'trash'}

def add_class_name(df,col):
    df[col]=df[col].apply(lambda x: x[:re.search("\d",x).start()]+ '/'+x)
    return df

filenames=[]
categories=[]

for category in our_classes:
    file_name=os.listdir(img_path+our_classes[category])
    filenames=filenames+file_name
    categories=categories + [category]*len(file_name)

df=pd.DataFrame({'filename':filenames,'category':categories})
df=add_class_name(df,'filename')
df['category']=df['category'].astype(str)  #ADD THIS LINE WITh her.

##Getting the train/test split into arrays for our NN
train_datagen=image.ImageDataGenerator()
print('starting generator')
train_generator=train_datagen.flow_from_dataframe(df,img_path,x_col='filename',y_col='category',target_size=Image_dimension,class_mode='categorical',batch_size=64)

print(train_generator)

