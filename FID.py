import numpy as np
from numpy import trace, iscomplexobj, cov, asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence): #source: https://stackoverflow.com/questions/62916904/
                               #failed-copying-input-tensor-from-cpu-to-gpu-in-order-to-run-gatherve-dst-tensor
    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x


class FID_measure: #source: https://machinelearningmastery.com/
                   #how-to-implement-the-frechet-inception-distance-fid-from-scratch/
    
    def __init__(self, gen_images_, train_set_, labels_):
        self.desired_shape = (299, 299, 3)
        self.gen_images = gen_images_.astype('float32')
        self.train_set = train_set_.astype('float32')
        self.labels = labels_
        self.num_samples_each_class = 1000
        self.mnist_num_labels = 10
        self.train_set_shrinked = np.array()
    
    def sample_uniformly(self):
        sampled_indices = []
        for label in range(self.mnist_num_labels):
            label_indices = np.where(self.labels == label)[0]
            sampled_indices.extend(np.random.choice(label_indices, size=self.num_samples_each_class, replace=False))

        sampled_indices = np.array(sampled_indices)
        sampled_images = self.train_set[sampled_indices]
        sampled_labels = self.labels[sampled_indices]
        ## Shuffling (not obligatory)
        shuffle_indices = np.random.permutation(len(sampled_labels))
        sampled_images = sampled_images[shuffle_indices]
        self.train_set_shrinked = sampled_images
    
    
    def calculate_fid(self): 
        self.train_set_shrinked = self.scale_images(self.train_set_shrinked)
        self.gen_images = self.scale_images(self.gen_images)
        self.train_set_shrinked = preprocess_input(self.train_set_shrinked)
        self.gen_images = preprocess_input(self.gen_images)
        print('beginning of calc_fid')
        self.gen_images = DataGenerator(x_set = self.gen_images, batch_size = 32)
        self.train_set_shrinked = DataGenerator(x_set = self.train_set_shrinked, batch_size = 32)
        model = InceptionV3(include_top=False, pooling='avg', input_shape = self.desired_shape)
        # calculate activations
        act1 = model.predict(self.gen_images)
        print('gen_images solved')
        act2 = model.predict(self.train_set_shrinked)
        print('train_set solved')
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        print('sigma1 solved')
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        print('sigma2 solved')
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    
    def scale_images(self, images):
        images_list = list()
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, self.desired_shape, 0)
            # store
            images_list.append(new_image)
        return asarray(images_list)
    