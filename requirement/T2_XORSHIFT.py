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

def XORShift_real(x, y, nCluster=32, SEED=0):
	temp = ((x & 0xFFFF)<<16)+(y & 0xFFFF)
	temp ^= temp <<13;
	temp ^= temp >>7;
	temp ^= temp <<17;
	return temp%nCluster

@jit(nopython=True, cache=True)
def step1_fc(nhWeight, nwWeight, nCluster, SEED=0):
	cLabel = np.zeros((nhWeight, nwWeight))		#nothing

	for i in range(nhWeight):
		for i2 in range(nwWeight):
			#initialize
			centroid = (((i & 0xFFFF)*0x8521)+((i2 & 0xFFFF)*0x1252) + 0x3424 ) ^ 0x4920;
			
			#case1
			#centroid = ((i & 0xFFFF)<<16)+(i2 & 0xFFFF)
			#centroid ^= (centroid <<13) & 0xFFFFFFFF;
			#centroid ^= (centroid >>7) & 0xFFFFFFFF;
			#centroid ^= (centroid <<17) & 0xFFFFFFFF;
			
			#case2
			centroid = i *3266489917 + 374761393
			centroid = (centroid << 17) | (centroid >> 15);
			centroid += i2 * 3266489917

			centroid *= 668265263;
			centroid ^= centroid > 15;
			centroid *= 2246822519;
			centroid ^= centroid > 13;
			centroid *= 3266489917;
			centroid ^= centroid > 16;

			cLabel[i][i2] = centroid % nCluster;
	return cLabel;

@jit(nopython=True, cache=True)
def step1_conv(nhWeight, nwWeight, nciWeight, ncoWeight, nCluster, SEED=0):
	cLabel = np.zeros((nhWeight, nwWeight, nciWeight, ncoWeight))		#nothing

	for i in range(nhWeight):
		for i2 in range(nwWeight):
			for i3 in range(nciWeight):
				for i4 in range(ncoWeight):
					temp = ((i & 0xFFFFFFFF)<<48) + ((i2 & 0xFFFFFFFF)<<32) + ((i3 & 0xFFFFFFFF)<<16) + i2
					temp ^= temp <<13;
					temp ^= temp >>7;
					temp ^= temp <<17;
					cLabel[i][i2] = centroid % nCluster;

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

class XORShift(object):
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
						self.cLabel[i][i2] = XORShift_real(i, i2, self.nCluster_per_block, seed)+int(i/self.nhWeight_per_block)*self.nCluster_per_block;
					
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

