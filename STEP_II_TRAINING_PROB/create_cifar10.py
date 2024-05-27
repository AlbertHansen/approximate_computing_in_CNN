#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


# In[2]:


# settings
save_directory    = '/home/ubuntu/tensorflow_datasets/cifar10/'
apply_grayscale   = False
reduce_dimensions = False
normalise_data    = True
resize_method     = tf.image.ResizeMethod.NEAREST_NEIGHBOR
data_size_x       = 16
data_size_y       = data_size_x


# In[3]:


# Save Summary
with open(save_directory + 'configurations.txt', 'w') as f:
    f.write(f'save_directory    = {save_directory}\n')
    f.write(f'apply_grayscale   = {apply_grayscale}\n')
    f.write(f'reduce_dimensions = {reduce_dimensions}\n')
    f.write(f'normalise_data    = {normalise_data}\n')
    f.write(f'resize_method     = {resize_method}\n')
    f.write(f'data_size_x       = {data_size_x}\n')
    f.write(f'data_size_y       = {data_size_y}\n')


# In[4]:


# Dataset
cifar10 = tfds.builder('cifar10')

# Description of dataset
assert cifar10.info.features['image'].shape       == (32, 32, 3)
assert cifar10.info.features['label'].num_classes == 10
assert cifar10.info.splits['train'].num_examples  == 50000
assert cifar10.info.splits['test'].num_examples   == 10000

# Download and prepare the data
cifar10.download_and_prepare()
datasets = cifar10.as_dataset()


'''
# Load data and convert to grayscale
if apply_grayscale:
    for split in ['train', 'test']:
        datasets[split] = datasets[split].map(lambda item: {
            'image': tf.image.rgb_to_grayscale(item['image']),
            'label': item['label']
        })

# Reduce dimensionality
if reduce_dimensions:
    for split in ['train', 'test']:
        datasets[split] = datasets[split].map(lambda item: {
            'image': tf.image.resize(
                item['image'],
                [data_size_x, data_size_y],
                method=resize_method, 
            ),
            'label': item['label']
        })
'''
# Normalise
if normalise_data:
    # Take one example from the dataset
    for example in datasets['train'].take(1):
        image, label = example['image'], example['label']

    # Check maximal pixel value and normalise if above 1
    max_pixel_value = tf.reduce_max(image).numpy()
    if max_pixel_value > 1:    
        for split in ['train', 'test']:
            datasets[split] = datasets[split].map(lambda item: {
                'image': item['image'] / 255,
                'label': item['label']
            })


# Save the datasets
for split in ['train', 'test']:
    datasets[split].save(save_directory + split)


# %%
