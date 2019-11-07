import tensorflow as tf
import numpy as np
import sys
from numba import jit

from kmeans_1D import kmeans_cluster
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
'''
@jit(nopython=True, cache=True)
def step1(nhWeight, nwWeight):
	cLabel = -np.ones((nhWeight, nwWeight), dtype=int)		#nothing

	for i in range(nhWeight):
		for i2 in range(nwWeight):
			cLabel[i][i2] = XXhash_real(i, i2) #0:i, 1:i2

	return cLabel;
'''
@jit(nopython=True, cache=True)
def step1(nhWeight, nwWeight, nCluster, SEED=0):
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

class XXhash_kmeans(object):
	def __init__(self, cWeights, XXarray, XXnCluster=9600, nCluster=32, blocked=False, blocked_param=64, seed=0, max_iter=1000):
		self.nCluster = nCluster
	
		if blocked :
			self.nCluster_per_block = int(nCluster/blocked_param);
			self.XXnCluster_per_block = int(XXnCluster/blocked_param);
			print("ncluster/block is", self.nCluster_per_block);
			print("XXncluster/block is", self.XXnCluster_per_block);
			if cWeights.shape[0]%blocked_param == 0 :
				self.nhWeight_per_block = int(cWeights.shape[0]/blocked_param);
			else :
				self.nhWeight_per_block = int(cWeights.shape[0]/blocked_param)+1;
			print("nhWeight/block is", self.nhWeight_per_block);

		self.nhWeight = cWeights.shape[0]
		self.nwWeight = cWeights.shape[1]

		self.cWeights = cWeights
		self.cLabel = -np.ones((self.nhWeight, self.nwWeight), dtype=int)		#nothing
		#self.mask = mask 	#nothing


		self.before_cCentro = -np.ones(XXnCluster)						#nothing
		for i in range(self.nhWeight):
			for i2 in range(self.nwWeight):
				if self.before_cCentro[XXarray[i][i2]]==-1:
					self.before_cCentro[XXarray[i][i2]]=cWeights[i][i2];
				elif self.before_cCentro[XXarray[i][i2]]!=cWeights[i][i2]:
					print("ERROR!!!");
		#--------------------------
		if blocked :	
			#STEP1 : hash index change
			label_temp = []
			self.cCentro = []
			for i in range(blocked_param):
				if i> int(self.nhWeight/self.nhWeight_per_block) :
					break;

				print("{}-----{}".format(i*self.XXnCluster_per_block, (i+1)*self.XXnCluster_per_block))
				kmeans = kmeans_cluster(\
					self.before_cCentro[i*self.XXnCluster_per_block:(i+1)*self.nhWeight_per_block],\
					nCluster=self.nCluster_per_block,\
					max_iter=max_iter);
				label_temp.append(kmeans.label() + self.nCluster_per_block*i);
				print("WOWOWOWOWOW-----", label_temp)
				self.cCentro.append(kmeans.cCentro());
			
				for i2 in range(self.nwWeight):
					self.cLabel[i][i2] = XXhash_real(i, i2, self.nCluster_per_block, seed)+int(i/self.nhWeight_per_block)*self.nCluster_per_block;
					
			if len(label_temp)!= self.nCluter_per_block*(int(self.nhWeight/self.nhWeight_per_block)+1):
				print("ERROR")
				return;
		else :
			#STEP1 : hash index change
			self.cCentro = -np.ones(self.nCluster)						#nothing

			kmeans = kmeans_cluster(self.before_cCentro, nCluster=nCluster, max_iter=max_iter);
			label_temp = kmeans.label();
			#np.set_printoptions(threshold=sys.maxsize)
			#print("size of !!label is {}".format(XXarray)) 
			#print("size of !!label is {}".format(len(label_temp))) 
			#print("size of !!label is {}".format(label_temp)) 
			#print("size of !!cCentro is {}".format(len(kmeans.cCentro))) 
			#print("size of !!XXarray is {}".format(XXarray.shape)) 
			for i in range(self.nhWeight):
				for i2 in range(self.nwWeight):
					self.cLabel[i][i2] = label_temp[XXarray[i][i2]];
			print("size of !!label is {}".format(self.cLabel)) 

			#STEP2 : get centroid by cLabel
			self.cCentro = kmeans.centro();
			print("before---------------------- is {}".format(self.before_cCentro)) 
			print("after----------------------- is {}".format(self.cCentro)) 

		#---------------------------

		
		

		#print("------------------------------------------------")
		#print("cCentro is ", self.cCentro);
		#print("cCentro_sum is ", centroid_sum);
		#self.cCentro = self.cCentro / centroid_sum
		#print("cCentro is ", self.cCentro);
		#print("------------------------------------------------")
		
		#self.cCentro = step2(self.nhWeight, self.nwWeight, self.nCluster).astype(int);

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

