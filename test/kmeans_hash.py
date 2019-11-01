import tensorflow as tf
import numpy as np
import sys
from numba import jit
#import time

#cWeights	: input weights!!
#clabel 	: return Label
#cCentro 	: return Centroid

#nhWeights	: # of weights h
#nwWeights	: # of weights w
#mask		: sparse
#nCluster	: # of centroids
#max_iter	: 1000
def XXhash(x, y):
	PRIME32_1 = int('0x9E3779B1', 16)
	PRIME32_2 = int('0x85EBCA77', 16)
	PRIME32_3 = int('0xC2B2AE3D', 16)
	PRIME32_4 = int('0x27D4EB2F', 16)
	PRIME32_5 = int('0x165667B1', 16)
	MASK = 0b11111

	seed = 0
	input_length = 0

	acc1 = seed + PRIME32_1 + PRIME32_2
	acc2 = seed + PRIME32_2
	acc3 = seed
	acc4 = seed - PRIME32_1

	lane1 = x
	lane2 = x+1
	lane3 = y
	lane4 = y+1

	acc1_w = acc1 + (lane1 * PRIME32_2)
	acc2_w = acc2 + (lane2 * PRIME32_2)
	acc3_w = acc3 + (lane3 * PRIME32_2)
	acc4_w = acc4 + (lane4 * PRIME32_2)

	acc = (acc1_w << 1) + (acc2_w << 7) +(acc3_w << 12) + (acc4_w << 18) + input_length

	centroid = (( ((acc ^ (acc>>15))*PRIME32_2)^(acc>>13)) * PRIME32_3) ^ (acc>>16)
	#centroid = int(centroid & MASK) % 30
	centroid = centroid % 30
	#print(bin(centroid))
	return centroid


class kmeans_hash_cluster(object):
	def __init__(self, cWeights, nCluster=30):
		self.nCluster = nCluster

		self.nhWeight = cWeights.shape[0]
		self.nwWeight = cWeights.shape[1]

		self.cWeights = cWeights
		self.cLabel = -np.ones((self.nhWeight, self.nwWeight), dtype=int)		#nothing
		self.cCentro = np.zeros(self.nCluster)						#nothing
		#self.mask = mask 	#nothing

		#index matrix
		#index = np.zeros((self.nhWeight,self.nwWeight, 2), dtype=int)
		#for i in range(self.nhWeight):
		#	for i2 in range(self.nwWeight):
		#		index[i][i2] = [i, i2]
	
		#hash index for each 
		#self.cLabel = hash(index)
		for i in range(self.nhWeight):
			for i2 in range(self.nwWeight):
				#self.cLabel[i][i2] = XXhash(index[i][i2][0], index[i][i2][1]) #0:i, 1:i2
				self.cLabel[i][i2] = XXhash(i, i2) #0:i, 1:i2

		#get centroid by cLabel
		centroid_sum = np.zeros(self.nCluster, dtype = int)
		for i in range(self.nhWeight):
			for i2 in range(self.nwWeight):
				self.cCentro[self.cLabel[i][i2]] += cWeights[i][i2]
				centroid_sum += 1

		self.cCentro = self.cCentro / centroid_sum

	def real_weight(self):
		return self.cWeights
	def weight(self):
		return self.cCentro[self.cLabel]
	def label(self, flatten=False):
		if flatten :
			return self.cLabel.flatten()
		else :
			return self.cLabel
	def centro(self):
		return self.cCentro
	def num_centro(self):
		return self.nCluster

