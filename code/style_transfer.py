from __future__ import absolute_import
from re import X
from tarfile import is_tarfile
from click import style
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, AvgPool2D
import numpy as np
import matplotlib.pyplot as plt
import os
from cnn_train_kento import Model, test
from get_data import get_data


class StyleTransfer:
	def __init__(self, model, train_model, input_image, style_image, feature_image):
		"""
		Initialize StyleTransfer function.

		:param model: The CNN model used to create dictionaries for latent tensors of style and feature images
		:param train_model: An identical, but separate instance of CNN model that is used to generate latents for the iterating image
		:input_image: A tensor representation of a randomized image to undergo iteration and generation of a new image (1, 256, 256, 3)
		:style_image: A tensor representation of the image that has the style you want to mimic (1, 256, 256, 3)
		:feature_image: A tensor representation of the image that has the features you want to mimic (1, 256, 256, 3)
		"""
		self.model = model # CNN model
		self.train_model = train_model # Model that training of image will use
		self.image = input_image # input image that is static --> to be changed
		self.style_image = style_image # image with style/texture
		self.feature_image = feature_image # image with features

		self.block_id = ['block1', 'block2', 'block3', 'block4', 'block5']

		# Dictionaries for latent tensors of reference style and features
		self.style_latents_dict = model.call(style_image, dense=False, style=True, is_training=False, feature=False)
		self.feature_latents_dict = model.call(feature_image, dense=False, style=False, is_training=False, feature=True)
		
		self.num_iter = 500 # Number of iterations #used to be 1000
		self.total_loss = 0 # Total loss (resets for every iteration)

		# Weights for Loss
		self.alpha = 1 # Feature
		self.beta = 1E7 # Style
		
		# Image optimization
		self.lr = 0.01 # learning rate
		self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)

	def train_step(self, input_image):
		"""
		A single training step that calculates the loss and applies the gradient to the input_image through backpropagation.
		This improves the input_image to incorporate the style of the style_image, and features of the feature_image.

		::param input_image: StyleTransfer.image
		"""
		with tf.GradientTape() as tape:
			clear_latents(self.train_model)
			x_style = self.train_model.call(input_image, dense=False, style=True, is_training=False, feature=False)
			x_feature = self.train_model.call(input_image, dense=False, style=False, is_training=False, feature=True)

			L_total = self.loss(x_style, x_feature)
		
		grads = tape.gradient(L_total, input_image)
		self.optimizer.apply_gradients([(grads, input_image)])
		self.image.assign(tf.clip_by_value(input_image, clip_value_min=0.0, clip_value_max=1.0)) # update image and make sure values are [0-1]


	def loss(self, x_style, x_feature):
		"""
		The loss function for style transfer.

		:param x_style: The dictionary holding the style latent-space representation of the input_image
		:param x_feature: The dictionary holding the feature latent-space representation of the input_image
		:return: Total loss (linear combination loss between style and features)
		"""
		# initialize style and feature loss per iteration
		L_style = 0
		L_feature = 0
		
		# cycle through each block id
		for id in self.block_id:
			# STYLE LOSS
			if id in self.style_latents_dict:

				A = self.style_latents_dict[id] # Latent for style_image at specified layer
				A = self.gram_matrix(A) # get Gram matrix of latent space of style_image
				F = x_style[id] # Style latents for input_image

				_, h, w, M = np.shape(F) # dimensions of style latents
				N = h*w 
				G = self.gram_matrix(tf.convert_to_tensor(F)) # get Gram matrix for F

				# multiply by 1/5 so for each block
				# Style Loss
				L_style += 1/5*(1./(4*N**2*M**2))*tf.reduce_sum(tf.pow(G-A, 2))

			# FEATURE LOSS
			if id in self.feature_latents_dict:
				
				P = self.feature_latents_dict[id] # Latent for feature image
				F = x_feature[id] # Feature latent for image_input

				# Feature Loss
				L_feature = 0.5*tf.reduce_sum(tf.pow(F-P, 2))
		# Linear combination of Feature and Style Losses
		L_total = self.alpha*L_feature + self.beta*L_style
		# print(L_feature.numpy(), L_style.numpy(), L_total.numpy())
		self.total_loss = L_total # update total loss attribute
		return L_total


	def gram_matrix(self, input_tensor):
		"""
		Get the Gram Matrix of the given tensor.
		https://www.tensorflow.org/tutorials/generative/style_transfer
		:param: input_tensor: any tensor
		:return: Gram matrix of input_tensor
		"""
		result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
		input_shape = tf.shape(input_tensor)
		num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
		return result/num_locations


def clear_latents(model):
	"""
	Clear the latent dictionaries for the given model.
	:param model: model you want to clear latent dictionaries of
	:return: None
	"""
	model.feature_latent = {}
	model.latents = {}

def visualize_image(image_tensor):
	"""
	Output image on device.
	:param image_tensor: the tensor representation of an image you want to visualize (1, 256, 256, 3)
	:return: None, but will just display the image on your screen using PIL
	"""
	image = np.array(image_tensor).reshape((256,256,3))
	image = tf.keras.preprocessing.image.array_to_img(image, scale=True) # 'scale' multiplies the array by 255
	image.show()
	pass

def visualize_loss(loss):
	x = np.arange(1, len(loss)+1)
	plt.xlabel('i\'th Batch')
	plt.ylabel('Loss Value')
	plt.title('Loss per Iteration')
	plt.plot(x, loss)
	plt.show()

def main():
	"""
	The function that runs the script and trains the input_image
	"""
	model_init = Model() # old model with dense layers
	model = Model() # just the CNN (no dense layers)
	train_model = Model() # just the CNN (no dense layers)

	model_init(tf.zeros((1,256,256,3)), dense=True)
	model(tf.zeros((1,256,256,3)), dense=False) 
	train_model(tf.zeros((1,256,256,3)), dense=False) 
	model_init.load_weights('../checkpoints/alex_best_weights.h5') # load weights into old model
	# apply relevant weights to the new models with only convolution layers
	for i, m in enumerate(model_init.layers[:-4]): # exclude layers from flatten to dense3
		model.layers[i].set_weights(model_init.layers[i].get_weights())
		train_model.layers[i].set_weights(model_init.layers[i].get_weights())
	del model_init # delete the old model with dense layers to free memory
	model.summary() # show summary of model

	# load data set with features
	# currently batches by 1 and no shuffle to get the same initial image
	train_dataset, test_dataset = get_data('../ISIC_data/Train/', '../ISIC_data/Test/', 
											batch_sz=1, shuffle=False, 
											image_sz=(256,256)) # try changing to (600,450) on GPU
	test_acc = test(model, test_dataset) # check the accuracy of the model
	print(f'Initial accuracy of model on classification: {test_acc*100} %')

	##### FEATURE PREPROCESS #####
	counter = 0
	for feature_image, labels in test_dataset:
		if counter == 16*3+2:
			feature_image /= 255.
			break
		counter += 1

	##### STYLE PREPROCESS #####
	style_image = tf.keras.preprocessing.image.load_img('../preprocessed_images/img13.jpg', target_size=(256, 256))
	style_image = tf.Variable(tf.reshape(tf.keras.preprocessing.image.img_to_array(style_image), (1, 256, 256, 3))/255., trainable=False)

	##### GENERATE STATIC IMAGE #####
	# generated_image = tf.Variable(tf.random.uniform((1,256,256,3), minval=0, maxval=1), name='Generated_Image', trainable=True)
	generated_image = tf.Variable(style_image.numpy(), trainable=True)

	#### START STYLE TRANSFER ####
	styletransfer = StyleTransfer(model, train_model, generated_image, style_image, feature_image) # instantiate model
	
	loss_list = []

	visualize_image(styletransfer.image) # visualize initial input
	for i in range(styletransfer.num_iter):
		styletransfer.train_step(styletransfer.image)
		if i % 100 == 0: print(f'Loss at iteration {i}: {styletransfer.total_loss}')
		loss_list.append(styletransfer.total_loss)

	visualize_image(styletransfer.image) # visualize output
	visualize_loss(loss_list)


if __name__ == '__main__':
	main()