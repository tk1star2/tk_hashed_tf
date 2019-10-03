import tensorflow as tf
import numpy as np
import sys

#cWeights	: input weights!!
#clabel 	: return Label
#cCentro 	: return Centroid

#nhWeights	: # of weights h
#nwWeights	: # of weights w
#mask		: sparse
#nCluster	: # of centroids
#max_iter	: 1000
class kmeans_cluster(object):
	def __init__(self, cWeights, nCluster=30, max_iter=1000):
		self.nCluster = nCluster
		self.max_iter = max_iter


		self.maxWeight= max(map(max, cWeights))
		self.minWeight= min(map(min, cWeights))
		self.nhWeight = cWeights.shape[0]
		self.nwWeight = cWeights.shape[1]

		self.cWeights = cWeights
		self.cLabel = np.zeros((self.nhWeight, self.nwWeight), dtype=int)		#nothing
		self.cCentro = np.zeros(self.nCluster)						#nothing
		#self.mask = mask 	#nothing
		'''
		for i in range(nWeights):
			if mask[i]:
				if cWeights[i] > maxWeight:
					maxWeight = cWeight[i]
				if cWeights[i] < minWeight:
					minWeight = cWeight[i]
		'''
		for k in range(self.nCluster):
			self.cCentro[k] = self.minWeight + (self.maxWeight - self.minWeight) * k / (self.nCluster-1)

		for i in range(self.nhWeight):
			for i2 in range(self.nwWeight):
				self.cLabel[i][i2] = -1

		cDistance 		= np.zeros((self.nhWeight, self.nwWeight))
		cClusterSize	= np.zeros(self.nCluster, dtype = int)
		pCentroPos 		= np.zeros(self.nCluster)
		pClusterSize 	= np.zeros(self.nCluster, dtype = int)
		ptrC			= np.zeros(self.nCluster)
		ptrS			= np.zeros(self.nCluster, dtype = int)


		mPreDistance = sys.float_info.max
		mCurDistance = 0.0
		for iter in range(max_iter):
			print("tk.kmeans iteration : ", iter)
			# check convergence
			if abs(mPreDistance-mCurDistance) / mPreDistance < 0.01 :
				break
			mPreDistance = mCurDistance
			mCurDistance = 0.0

			# select nearest cluster
			for i in range(self.nhWeight):
				for i2 in range(self.nwWeight):
					mindistance = sys.float_info.max
					clostCluster = -1
					for k in range(self.nCluster):
						distance = abs(self.cWeights[i][i2] - self.cCentro[k])
						if distance < mindistance :
							mindistance = distance
							clostCluster = k
					cDistance[i][i2] = mindistance
					self.cLabel[i][i2] = clostCluster;

			# calculate new distance
			for i in range(self.nhWeight):
				for i2 in range(self.nwWeight):
					mCurDistance += cDistance[i][i2]

			# generate new centroids
			for k in range(self.nCluster):
					ptrC[k] = float(0)
					ptrS[k] = 0
	
			for i in range(self.nhWeight):
				for i2 in range(self.nwWeight):
					ptrC[self.cLabel[i][i2]] += self.cWeights[i][i2]
					ptrS[self.cLabel[i][i2]] += 1

			for k in range(self.nCluster):
					pCentroPos[k] = ptrC[k]
					pClusterSize[k] = ptrS[k]

			
			for k in range(self.nCluster):
					self.cCentro[k] = pCentroPos[k]
					cClusterSize[k] = pClusterSize[k]
					self.cCentro[k] /= cClusterSize[k]
'''
	def step1(self):
		mindistance = sys.float_info.max
		clostCluster = -1
		for k in range(self.nCluster):
			distance = abs(self.cWeights[i][i2] - self.cCentro[k])
			if distance < mindistance :
				mindistance = distance
				clostCluster = k
		cDistance[i][i2] = mindistance
		self.cLabel[i][i2] = clostCluster;'''
			

	def weight(self):
		return self.cWeights
	#@property?
	def label(self):
		return self.cLabel
	def centro(self):
		return self.cCentro
	def num_centro(self):
		return self.nCluster
