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

@jit(nopython=True, cache=True)
def step1(nhWeight, nwWeight, nCluster, SEED=0):

	return cLabel;


class Pruned(object):
	def __init__(self, cWeights, XXarray, XXnCluster=9600, blocked=False, blocked_param=64, sparse_ratio=0.1):
		self.nCluster = XXnCluster
	
		self.nhWeight = cWeights.shape[0]
		self.nwWeight = cWeights.shape[1]
		if blocked :
			self.nBlock=blocked_param
			self.nCluster_per_block = int(self.nCluster/blocked_param);
			self.sparse_ratio = self.nCluster_per_block * sparse_ratio;
			print("ncluster/block is", self.nCluster_per_block);
			if cWeights.shape[0]%blocked_param == 0 :
				self.nhWeight_per_block = int(self.nhWeight/blocked_param);
				self.nBlock = int(blocked_param);
			else :
				self.nhWeight_per_block = int(self.nhWeight/blocked_param)+1;
				self.nBlock = int(self.nhWeight/self.nhWeight_per_block)+1;
				#self.nhWeight_no_exact = True;
			print("nhWeight/block is", self.nhWeight_per_block);
			print("!!!!!!!!!!!nBlock!!!!! see this !!!{}".format(self.nBlock))
		else :
			self.sparse_ratio = self.nCluster *sparse_ratio

		print("!!!!!!!!sparse ratio!!!!!!!! see this !!!{}".format(self.sparse_ratio))

		self.cWeights = cWeights
		#self.cLabel = -np.ones((self.nhWeight, self.nwWeight), dtype=int)		#nothing
		self.cLabel = XXarray		#nothing
		#self.mask = mask 	#nothing


		self.cCentro = -np.ones(self.nCluster)						#nothing
		self.cMask = -np.ones(self.nCluster)						#nothing
		for i in range(self.nhWeight):
			for i2 in range(self.nwWeight):
				if self.cCentro[XXarray[i][i2]]==-1:
					self.cCentro[XXarray[i][i2]]=cWeights[i][i2];
				elif self.cCentro[XXarray[i][i2]]!=cWeights[i][i2]:
					print("ERROR!!!");
		#--------------------------
		count = 0;
		count_real = 0;
		if blocked :	
			print("blocked!!!")
			for i in range(self.nBlock):
				sorted_index = np.argsort(abs(self.cCentro[i*self.nCluster_per_block:(i+1)*self.nCluster_per_block]), axis=0)
				#print("tk check:", i*self.nCluster_per_block)
				for i2 in range(self.nCluster_per_block):
					if i2 < self.sparse_ratio :
						self.cMask[i*self.nCluster_per_block + sorted_index[i2]]=1;
					else :
						count += 1;
						self.cMask[i*self.nCluster_per_block + sorted_index[i2]]=0;
		else :
			sorted_index = np.argsort(abs(self.cCentro), axis=0);

			print("tk check:", sorted_index)
			for i in range(self.nCluster):
				if i < self.sparse_ratio :
					self.cMask[sorted_index[i]]=1;
				else :
					count += 1;
					self.cMask[sorted_index[i]]=0;
					
		self.cRealWeight = -np.ones((self.nhWeight, self.nwWeight), dtype=float)		#nothing
		for i in range(self.nhWeight):
			for i2 in range(self.nwWeight):
				self.cRealWeight[i][i2] = self.cCentro[self.cLabel[i][i2]] * float(self.cMask[self.cLabel[i][i2]])
				if self.cRealWeight[i][i2]==0 :
					count_real += 1;
		print("!!!!!!!wantto!!!!!!!!!! see this !!!{}".format(count/(self.nCluster_per_block*self.nBlock)))
		print("!!!!!!!!!but!!!!!! see this !!!{}".format(count_real/(self.nhWeight * self.nwWeight)))

			#np.set_printoptions(threshold=sys.maxsize)
			#print("size of !!label is {}".format(XXarray)) 
			#print("size of !!label is {}".format(len(label_temp))) 
			#print("size of !!label is {}".format(label_temp)) 
			#print("size of !!cCentro is {}".format(len(kmeans.cCentro))) 
			#print("size of !!XXarray is {}".format(XXarray.shape)) 


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
		return self.cRealWeight
	def label(self, flatten=False):
		if flatten :
			return self.cLabel.flatten()
		else :
			return self.cLabel
	def centro(self):
		return self.cCentro
	def mask(self):
		return self.cMask
	def num_centro(self):
		return self.nCluster
	def num_block(self):
		return self.nBlock

