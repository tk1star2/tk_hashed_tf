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
	
# select nearest cluster
@jit(nopython=True, cache=True)
def step1(cWeights, nWeight, cCentro):
	cDistance = np.zeros(nWeight)
	cLabel = np.zeros(nWeight)
	for i in range(nWeight):
		distance = np.absolute(cWeights[i] - cCentro)
		cDistance[i] = min(distance)
		cLabel[i] = np.argmin(distance)

	return cDistance, cLabel

class kmeans_cluster(object):
	def __init__(self, cWeights, nCluster=30, max_iter=1000):
		self.nCluster = nCluster
		self.max_iter = max_iter


		self.maxWeight= max(cWeights)
		self.minWeight= min(cWeights)

		if cWeights.ndim != 1:
			print("kmeans clustering ERROR");
			return;

		self.nWeight = len(cWeights)

		self.cWeights = cWeights
		self.cLabel = -np.ones(self.nWeight, dtype=int)		#nothing
		self.cCentro = np.zeros(self.nCluster)						#nothing
		#self.mask = mask 	#nothing
		#print("check!!!!!!!!!!!! cLabel size is {}, cCentro size is {}".format(self.nWeight, self.nCluster))

		for k in range(self.nCluster):
			self.cCentro[k] = self.minWeight + (self.maxWeight - self.minWeight) * k / (self.nCluster-1)

		ptrC			= np.zeros(self.nCluster)				# value of sum centroids
		ptrS			= np.zeros(self.nCluster, dtype = int)	# num of all centroids


		mPreDistance = sys.float_info.max
		mCurDistance = 0.0
		for iter in range(max_iter):
			#print("tk.kmeans iteration : ", iter)
			# check convergence
			if abs(mPreDistance-mCurDistance) / mPreDistance < 0.01 :
				break
			mPreDistance = mCurDistance

			# select nearest cluster of points
			#start_time = time.time()
			cDistance, cLabel = step1(self.cWeights, self.nWeight, self.cCentro)
			self.cLabel = cLabel.astype(np.int);
			#print("-------------------endtime:{}---------------------".format(time.time()-start_time))
			# calculate new distance
			mCurDistance = sum(cDistance)

			# generate new centroids
			ptrC = np.zeros(self.nCluster)
			ptrS = np.zeros(self.nCluster, dtype = int)

			for i in range(self.nWeight):
				ptrC[self.cLabel[i]] += self.cWeights[i]
				ptrS[self.cLabel[i]] += 1

			# reduction		
			self.cCentro = ptrC
			for i in range(self.nCluster):
				if ptrS[i] != 0:
					self.cCentro[i] = self.cCentro[i] /  ptrS[i]

	def real_weight(self):
		return self.cWeights
	def weight(self):
		return self.cCentro[self.cLabel]
	def label(self):
		return self.cLabel
	def centro(self):
		return self.cCentro
	def num_centro(self):
		return self.nCluster

