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
def XXhash_real_fc(x, y, nCluster=32, SEED=0):
	'''
	PRIME32_1 = int('0x9E3779B1', 16)
	PRIME32_2 = int('0x85EBCA77', 16)
	PRIME32_3 = int('0xC2B2AE3D', 16)
	PRIME32_4 = int('0x27D4EB2F', 16)
	PRIME32_5 = int('0x165667B1', 16)
	#SEED = int('0x00000000', 16)
	#inputLength = nCluster;
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
	'''
	#case2
	# mix x & y
	centroid = x *3266489917 + 374761393
	centroid = (centroid << 17) | (centroid >> 15);
	centroid += y * 3266489917

	# stir centroid
	centroid *= 668265263;
	centroid ^= centroid > 15;
	centroid *= 2246822519;
	centroid ^= centroid > 13;
	centroid *= 3266489917;
	centroid ^= centroid > 16;

	#print(bin(centroid))
	#print(centroid % nCluster);
	return centroid % nCluster;

def XXhash_real_conv(i, i2, i3, i4, nCluster=32, SEED=0):
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
			'''
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
			'''
			#case2
			# mix x & y
			centroid = i *3266489917 + 374761393
			centroid = (centroid << 17) | (centroid >> 15);
			centroid += i2 * 3266489917

			# stir centroid
			centroid *= 668265263;
			centroid ^= centroid > 15;
			centroid *= 2246822519;
			centroid ^= centroid > 13;
			centroid *= 3266489917;
			centroid ^= centroid > 16;

			cLabel[i][i2] = centroid % nCluster;

	return cLabel;
@jit(nopython=True, cache=True)
def step1_conv(ncoWeight, nciWeight, nwWeight, nhWeight, nCluster, SEED=0):
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

	for i in range(ncoWeight):
		for i2 in range(nciWeight):
			for i3 in range(nwWeight):
				for i4 in range(nhWeight):
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

					cLabel[i4][i3][i2][i] = centroid % nCluster;

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
		if 64 < blocked_param:
			blocked_param = 64;

		print("Tfcheck !!!",blocked , "------ ", blocked_param, "--------", Conv_FC );
		if Conv_FC=="FC" :
			print("!!!!Tfc !!!",blocked , "------ ", blocked_param );
			print("--------", Conv_FC , "------------");
			if blocked :
				if cWeights.shape[0] < blocked_param:
					blocked_param = cWeights.shape[0];
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
						self.cLabel[i][i2] = XXhash_real_fc(i, i2, self.nCluster_per_block, seed)+int(i/self.nhWeight_per_block)*self.nCluster_per_block;
					
			else :
				self.cLabel = step1_fc(self.nhWeight, self.nwWeight, self.nCluster, seed).astype(int);

			#STEP2 : get centroid by cLabel
		
			centroid_sum = np.zeros(self.nCluster, dtype = int)
			for i in range(self.nhWeight):
				for i2 in range(self.nwWeight):
					self.cCentro[self.cLabel[i][i2]] += self.cWeights[i][i2]
					centroid_sum[self.cLabel[i][i2]] += 1

			for i in range(self.nCluster):
				if centroid_sum[i] != 0 :
					self.cCentro[i] = self.cCentro[i] / centroid_sum[i]
		
			#self.cCentro = step2(self.nhWeight, self.nwWeight, self.nCluster).astype(int);
		else : #-----------------------------------------------------------------------------
			print("!!!!Tfc !!!",blocked , "------ ", blocked_param );
			print("--------", Conv_FC , "------------");
			if blocked :
				if cWeights.shape[3] < blocked_param:
					blocked_param = cWeights.shape[3];
				self.nCluster_per_block = int(nCluster/blocked_param);
				print("ncluster/block is", self.nCluster_per_block);
				# TODO: didnt check (cWeights.shape[3] / blocked_param) ==0 is wrong	
				if cWeights.shape[3]%blocked_param == 0 :
					self.ncoWeight_per_block = int(cWeights.shape[3]/blocked_param);
				else :
					self.ncoWeight_per_block = int(cWeights.shape[3]/blocked_param)+1;
				print("ncoWeight/block is", self.ncoWeight_per_block);

			# 3, 3, 1, 32
			print("cWeights.shape is {}, {}, {}, {}".format(cWeights.shape[0],cWeights.shape[1],cWeights.shape[2],cWeights.shape[3]))
			self.nhWeight = cWeights.shape[0] #3
			self.nwWeight = cWeights.shape[1] #3
			self.nciWeight = cWeights.shape[2] #1
			self.ncoWeight = cWeights.shape[3] #32

			self.cWeights = cWeights
			self.cLabel = -np.ones((self.nhWeight, self.nwWeight, self.nciWeight, self.ncoWeight), dtype=int)		#nothing
			self.cCentro = np.zeros(self.nCluster)						#nothing
			#self.mask = mask 	#nothing
				
			#STEP1 : hash index for each 
			if blocked :
				for i in range(self.ncoWeight):
					for i2 in range(self.nciWeight):
						for i3 in range(self.nwWeight):
							for i4 in range(self.nhWeight):
								self.cLabel[i4][i3][i2][i] = XXhash_real_conv(i, i2, i3, i4, self.nCluster_per_block, seed)+int(i/self.ncoWeight_per_block)*self.nCluster_per_block;
					
			else :
				self.cLabel = step1_conv(self.ncoWeight, self.nciWeight, self.nwWeight, self.nhWeight, self.nCluster, seed).astype(int);

			#STEP2 : get centroid by cLabel
			centroid_sum = np.zeros(self.nCluster, dtype = int)
			for i in range(self.ncoWeight):
				for i2 in range(self.nciWeight):
					for i3 in range(self.nwWeight):
						for i4 in range(self.nhWeight):
							self.cCentro[self.cLabel[i4][i3][i2][i]] += self.cWeights[i4][i3][i2][i]
							centroid_sum[self.cLabel[i4][i3][i2][i]] += 1

			for i in range(self.nCluster):
				if centroid_sum[i] != 0 :
					self.cCentro[i] = self.cCentro[i] / centroid_sum[i]

		print("==============let's see=================");
		print("nCluster is ", nCluster)
		print("centro 0 is ", nCluster - np.count_nonzero(self.cCentro))
		print("0 is ", nCluster - np.count_nonzero(centroid_sum))
		print("0 percentange is ", (nCluster - np.count_nonzero(centroid_sum))/nCluster)
		print("nonzero is ", np.count_nonzero(centroid_sum))
		print("==============let's see=================");
		temp_t = 1
		#while 1 :
		#	if np.count_nonzero(self.cCentro==temp_t
		#	print("{} is ", np.count_nonzero(self.cCentro == 1))
		#	print("{} percentange is ", (np.count_nonzero(self.cCentro==1))/nCluster)
			
		print("1 is ", np.count_nonzero(centroid_sum == 1))
		print("1 percentange is ", (np.count_nonzero(centroid_sum==1))/nCluster, "or", (np.count_nonzero(centroid_sum==1))/np.count_nonzero(centroid_sum))
		print("2 is ", np.count_nonzero(centroid_sum == 2))
		print("2 percentange is ", (np.count_nonzero(centroid_sum==2))/nCluster, "or", (np.count_nonzero(centroid_sum==2))/np.count_nonzero(centroid_sum))
		print("3 is ", np.count_nonzero(centroid_sum == 3))
		print("3 percentange is ", (np.count_nonzero(centroid_sum==3))/nCluster, "or", (np.count_nonzero(centroid_sum==3))/np.count_nonzero(centroid_sum))
		print("4 is ", np.count_nonzero(centroid_sum == 4))
		print("4 percentange is ", (np.count_nonzero(centroid_sum==4))/nCluster, "or", (np.count_nonzero(centroid_sum==4))/np.count_nonzero(centroid_sum))
		print("5 is ", np.count_nonzero(centroid_sum == 5))
		print("5 percentange is ", (np.count_nonzero(centroid_sum==5))/nCluster, "or", (np.count_nonzero(centroid_sum==5))/np.count_nonzero(centroid_sum))
		print("6 is ", np.count_nonzero(centroid_sum == 6))
		print("6 percentange is ", (np.count_nonzero(centroid_sum==6))/nCluster, "or", (np.count_nonzero(centroid_sum==6))/np.count_nonzero(centroid_sum))
		print("7 is ", np.count_nonzero(centroid_sum == 7))
		print("7 percentange is ", (np.count_nonzero(centroid_sum==7))/nCluster, "or", (np.count_nonzero(centroid_sum==7))/np.count_nonzero(centroid_sum))
		print("8 is ", np.count_nonzero(centroid_sum == 8))
		print("8 percentange is ", (np.count_nonzero(centroid_sum==8))/nCluster, "or", (np.count_nonzero(centroid_sum==8))/np.count_nonzero(centroid_sum))
		print("9 is ", np.count_nonzero(centroid_sum == 9))
		print("9 percentange is ", (np.count_nonzero(centroid_sum==9))/nCluster, "or", (np.count_nonzero(centroid_sum==9))/np.count_nonzero(centroid_sum))
		print("10 is ", np.count_nonzero(centroid_sum == 10))
		print("10 percentange is ", (np.count_nonzero(centroid_sum==10))/nCluster, "or", (np.count_nonzero(centroid_sum==10))/np.count_nonzero(centroid_sum))

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

