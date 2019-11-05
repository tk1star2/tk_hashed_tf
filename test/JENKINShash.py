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

def JENKINS_real(x, y, nCluster=32):
	PRIME32_1 = int('0x9E3779B1', 16)
	PRIME32_2 = int('0x85EBCA77', 16)
	PRIME32_3 = int('0xC2B2AE3D', 16)
	PRIME32_4 = int('0x27D4EB2F', 16)
	PRIME32_5 = int('0x165667B1', 16)
	SEED = int('0x00000000', 16)
	inputLength = nCluster;
	MASK = 0b11111

	acc1 = SEED + PRIME32_1 + PRIME32_2;
	acc2 = SEED + PRIME32_2;
	acc3 = SEED;
	acc4 = SEED - PRIME32_1;

	lane1 = x;
	lane2 = x+1;
	lane3 = y;
	lane4 = y+1;

	acc1_w = acc1 + (lane1 * PRIME32_2)
	acc2_w = acc2 + (lane2 * PRIME32_2)
	acc3_w = acc3 + (lane3 * PRIME32_2)
	acc4_w = acc4 + (lane4 * PRIME32_2)

	acc = (acc1_w << 1) + (acc2_w << 7) +(acc3_w << 12) + (acc4_w << 18) + inputLength

	centroid = (( ((acc ^ (acc>>15))*PRIME32_2)^(acc>>13)) * PRIME32_3) ^ (acc>>16)
	#print(bin(centroid))
	#print(centroid % nCluster);
	return centroid % nCluster;

@jit(nopython=True, cache=True)
def step1(nhWeight, nwWeight):
	cLabel = -np.ones((nhWeight, nwWeight), dtype=int)		#nothing

	for i in range(nhWeight):
		for i2 in range(nwWeight):
			cLabel[i][i2] = JENKINS_real(i, i2) #0:i, 1:i2

	return cLabel;

class JENKINShash(object):
	def __init__(self, cWeights, nCluster=32):
		self.nCluster = nCluster

		self.nhWeight = cWeights.shape[0]
		self.nwWeight = cWeights.shape[1]

		self.cWeights = cWeights
		self.cLabel = -np.ones((self.nhWeight, self.nwWeight), dtype=int)		#nothing
		self.cCentro = np.zeros(self.nCluster)						#nothing
		#self.mask = mask 	#nothing

		#STEP1 : hash index for each 
		for i in range(self.nhWeight):
			for i2 in range(self.nwWeight):
				self.cLabel[i][i2] = JENKINS_real(i, i2, self.nCluster) #0:i, 1:i2
		#self.cLabel = step1(self.nhWeight, self.nwWeight);

		#STEP2 : get centroid by cLabel
		centroid_sum = np.zeros(self.nCluster, dtype = int)
		for i in range(self.nhWeight):
			for i2 in range(self.nwWeight):
				self.cCentro[self.cLabel[i][i2]] += self.cWeights[i][i2]
				centroid_sum[self.cLabel[i][i2]] += 1
		

		print("------------------------------------------------")
		print("cCentro is ", self.cCentro);
		print("cCentro_sum is ", centroid_sum);
		#self.cCentro = self.cCentro / centroid_sum
		for i in range(self.nCluster):
			if self.cCentro[i] != 0 :
				self.cCentro[i] = self.cCentro[i] / centroid_sum[i]
		print("cCentro is ", self.cCentro);
		print("------------------------------------------------")

		#STEP3 : weight update
		#for i in range(self.nhWeight):
		#	for i2 in range(self.nwWeight):
		#		self.cWeights[i][i2] = self.cCentro[self.cLabel[i][i2]]

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

