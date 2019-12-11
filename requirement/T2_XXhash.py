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

def XXhash_real(x, y, nCluster=32, SEED=0):
	PRIME32_1 = int('0x9E3779B1', 16)
	PRIME32_2 = int('0x85EBCA77', 16)
	PRIME32_3 = int('0xC2B2AE3D', 16)
	PRIME32_4 = int('0x27D4EB2F', 16)
	PRIME32_5 = int('0x165667B1', 16)
	#SEED = int('0x00000000', 16)
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
def step1_fc(nhWeight, nwWeight, nCluster, SEED=0):
	PRIME32_1 = 2654435761
	PRIME32_2 = 2246822519
	PRIME32_3 = 3266489917
	PRIME32_4 = 668265263
	PRIME32_5 = 374761393
	#PRIME32_1 = int('0x9E3779B1', 16)
	#PRIME32_2 = int('0x85EBCA77', 16)
	#PRIME32_3 = int('0xC2B2AE3D', 16)
	#PRIME32_4 = int('0x27D4EB2F', 16)
	#PRIME32_5 = int('0x165667B1', 16)
	#SEED = int('0x00000000', 16)
	inputLength = nCluster;
	cLabel = np.zeros((nhWeight, nwWeight))		#nothing

	for i in range(nhWeight):
		for i2 in range(nwWeight):
			acc1 = SEED + PRIME32_1 + PRIME32_2;
			acc2 = SEED + PRIME32_2;
			acc3 = SEED;
			acc4 = SEED - PRIME32_1;

			lane1 = i;
			lane2 = i+1;
			lane3 = i2;
			lane4 = i2+1;

			acc1_w = acc1 + (lane1 * PRIME32_2)
			acc2_w = acc2 + (lane2 * PRIME32_2)
			acc3_w = acc3 + (lane3 * PRIME32_2)
			acc4_w = acc4 + (lane4 * PRIME32_2)

			acc = (acc1_w << 1) + (acc2_w << 7) +(acc3_w << 12) + (acc4_w << 18) + inputLength

			centroid = (( ((acc ^ (acc>>15))*PRIME32_2)^(acc>>13)) * PRIME32_3) ^ (acc>>16)

			cLabel[i][i2] = centroid % nCluster;

	return cLabel;
@jit(nopython=True, cache=True)
def step1_conv(nhWeight, nwWeight, nciWeight, ncoWeight, nCluster, SEED=0):
	PRIME32_1 = 2654435761
	PRIME32_2 = 2246822519
	PRIME32_3 = 3266489917
	PRIME32_4 = 668265263
	PRIME32_5 = 374761393
	#PRIME32_1 = int('0x9E3779B1', 16)
	#PRIME32_2 = int('0x85EBCA77', 16)
	#PRIME32_3 = int('0xC2B2AE3D', 16)
	#PRIME32_4 = int('0x27D4EB2F', 16)
	#PRIME32_5 = int('0x165667B1', 16)
	#SEED = int('0x00000000', 16)
	inputLength = nCluster;
	cLabel = np.zeros((nhWeight, nwWeight, nciWeight, ncoWeight))		#nothing

	for i in range(nhWeight):
		for i2 in range(nwWeight):
			for i3 in range(nciWeight):
				for i4 in range(ncoWeight):
					acc1 = SEED + PRIME32_1 + PRIME32_2;
					acc2 = SEED + PRIME32_2;
					acc3 = SEED;
					acc4 = SEED - PRIME32_1;

					lane1 = i;
					lane2 = i2;
					lane3 = i3;
					lane4 = i4;

					acc1_w = acc1 + (lane1 * PRIME32_2)
					acc2_w = acc2 + (lane2 * PRIME32_2)
					acc3_w = acc3 + (lane3 * PRIME32_2)
					acc4_w = acc4 + (lane4 * PRIME32_2)

					acc = (acc1_w << 1) + (acc2_w << 7) +(acc3_w << 12) + (acc4_w << 18) + inputLength

					centroid = (( ((acc ^ (acc>>15))*PRIME32_2)^(acc>>13)) * PRIME32_3) ^ (acc>>16)

					cLabel[i][i2][i3][i4] = centroid % nCluster;

	return cLabel;

@jit(nopython=True, cache=True)
def step2(nhWeight, nwWeight, nCluster):
	#STEP2 : get centroid by cLabel
	centroid_sum = np.zeros(self.nCluster, dtype = int)
	for i in range(self.nhWeight):
		for i2 in range(self.nwWeight):
			self.cCentro[self.cLabel[i][i2]] += self.cWeights[i][i2]
			centroid_sum[self.cLabel[i][i2]] += 1
	
	#print("------------------------------------------------")
	#print("cCentro is ", self.cCentro);
	#print("cCentro_sum is ", centroid_sum);
	#self.cCentro = self.cCentro / centroid_sum
	for i in range(self.nCluster):
		if self.cCentro[i] != 0 :
			self.cCentro[i] = self.cCentro[i] / centroid_sum[i]
	#print("cCentro is ", self.cCentro);

class XXhash(object):
	def __init__(self, cWeights, Conv_FC="FC", nCluster=32, blocked=False, blocked_param=64, seed=0):
		self.nCluster = nCluster
		if 64 < nCluster :
			nCluster = 64;

		print("Tfc !!!",blocked , "------ ", blocked_param, "--------", Conv_FC );
		if Conv_FC=="FC" :
			print("Tfc !!!",blocked , "------ ", blocked_param );
			if blocked :
				self.nCluster_per_block = int(nCluster/blocked_param);
				print("ncluster/block is", self.nCluster_per_block);
				if cWeights.shape[0]%blocked_param == 0 :
					self.nhWeight_per_block = int(cWeights.shape[0]/blocked_param);
				else :
					self.nhWeight_per_block = int(cWeights.shape[0]/blocked_param)+1;
				print("nhWeight/block is", self.nhWeight_per_block);

			self.nhWeight = cWeights.shape[0]
			self.nwWeight = cWeights.shape[1]

			self.cWeights = cWeights
			self.cLabel = -np.ones((self.nhWeight, self.nwWeight), dtype=int)		#nothing
			self.cCentro = np.zeros(self.nCluster)						#nothing
			#self.mask = mask 	#nothing

			#STEP1 : hash index for each 
			if blocked :
				for i in range(self.nhWeight):
					for i2 in range(self.nwWeight):
						self.cLabel[i][i2] = XXhash_real(i, i2, self.nCluster_per_block, seed)+int(i/self.nhWeight_per_block)*self.nCluster_per_block;
					
			else :
				self.cLabel = step1_fc(self.nhWeight, self.nwWeight, self.nCluster, seed).astype(int);

			#STEP2 : get centroid by cLabel
		
			centroid_sum = np.zeros(self.nCluster, dtype = int)
			for i in range(self.nhWeight):
				for i2 in range(self.nwWeight):
					self.cCentro[self.cLabel[i][i2]] += self.cWeights[i][i2]
					centroid_sum[self.cLabel[i][i2]] += 1

			for i in range(self.nCluster):
				if self.cCentro[i] != 0 :
					self.cCentro[i] = self.cCentro[i] / centroid_sum[i]
		
			#self.cCentro = step2(self.nhWeight, self.nwWeight, self.nCluster).astype(int);
		else :
			if blocked :
				self.nCluster_per_block = int(nCluster/blocked_param);
				print("ncluster/block is", self.nCluster_per_block);
				# TODO: didnt check (cWeights.shape[3] / blocked_param) ==0 is wrong	
				if cWeights.shape[3]%blocked_param == 0 :
					self.ncoWeight_per_block = int(cWeights.shape[3]/blocked_param);
				else :
					self.ncoWeight_per_block = int(cWeights.shape[3]/blocked_param)+1;
				print("nhWeight/block is", self.nhWeight_per_block);

			# 3, 3, 1, 32
			print("cWeights.shape is {}, {}, {}, {}".format(cWeights.shape[0],cWeights.shape[1],cWeights.shape[2],cWeights.shape[3]))
			self.nhWeight = cWeights.shape[0]
			self.nwWeight = cWeights.shape[1]
			self.nciWeight = cWeights.shape[2]
			self.ncoWeight = cWeights.shape[3]

			self.cWeights = cWeights
			self.cLabel = -np.ones((self.nhWeight, self.nwWeight, self.nciWeight, self.ncoWeight), dtype=int)		#nothing
			self.cCentro = np.zeros(self.nCluster)						#nothing
			#self.mask = mask 	#nothing
				
			#STEP1 : hash index for each 
			if blocked :
				for i in range(self.nhWeight):
					for i2 in range(self.nwWeight):
						for i3 in range(self.nciWeight):
							for i4 in range(self.ncoWeight):
								self.cLabel[i][i2] = XXhash_real(i, i2, self.nCluster_per_block, seed)+int(i/self.ncoWeight_per_block)*self.nCluster_per_block;
					
			else :
				self.cLabel = step1_conv(self.nhWeight, self.nwWeight, self.nciWeight, self.ncoWeight, self.nCluster, seed).astype(int);

			#STEP2 : get centroid by cLabel
			centroid_sum = np.zeros(self.nCluster, dtype = int)
			for i in range(self.nhWeight):
				for i2 in range(self.nwWeight):
					for i3 in range(self.nciWeight):
						for i4 in range(self.ncoWeight):
							self.cCentro[self.cLabel[i][i2][i3][i4]] += self.cWeights[i][i2][i3][i4]
							centroid_sum[self.cLabel[i][i2][i3][i4]] += 1

			for i in range(self.nCluster):
				if self.cCentro[i] != 0 :
					self.cCentro[i] = self.cCentro[i] / centroid_sum[i]
			

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

