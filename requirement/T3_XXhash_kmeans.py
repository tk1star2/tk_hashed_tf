import tensorflow as tf
import numpy as np
import sys
from numba import jit

from T3_kmeans_1D import kmeans_cluster
#import time

#cWeights	: input weights!!
#clabel 	: return Label
#cCentro 	: return Centroid

#nhWeights	: # of weights h
#nwWeights	: # of weights w
#mask		: sparse
#nCluster	: # of centroids
#max_iter	: 1000

class XXhash_kmeans(object):
	def __init__(self, cWeights, XXarray, XXnCluster=9600, nCluster=2048, blocked=False, blocked_param=64, seed=0, max_iter=1000, Conv_FC="FC"):
		self.nCluster = nCluster

		if 64 < blocked_param:
			blocked_param=64;

		print("Tfcheck !!!",blocked , "------ ", blocked_param, "--------", Conv_FC );
		if Conv_FC=="FC" :
			print("!!!!Tfc !!!",blocked , "------ ", blocked_param );
			print("--------", Conv_FC , "------------");

			print(XXarray);
			self.nhWeight = cWeights.shape[0]
			self.nwWeight = cWeights.shape[1]
	
			if blocked :
				self.nCluster_per_block = int(nCluster/blocked_param);
				self.XXnCluster_per_block = int(XXnCluster/blocked_param);
				print("ncluster/block is", self.nCluster_per_block);
				print("XXncluster/block is", self.XXnCluster_per_block);
				if self.nhWeight%blocked_param == 0 :
					self.nhWeight_per_block = int(self.nhWeight/blocked_param);
					self.nBlock = int(blocked_param)
				else :
					self.nhWeight_per_block = int(self.nhWeight/blocked_param)+1;
					self.nBlock = int(self.nhWeight/self.nhWeight_per_block)+1
				print("nhWeight/block is", self.nhWeight_per_block);


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
				for i in range(self.nBlock):

					kmeans = kmeans_cluster(\
						self.before_cCentro[i*self.XXnCluster_per_block:(i+1)*self.XXnCluster_per_block],\
						nCluster=self.nCluster_per_block,\
						max_iter=max_iter);
					label_temp.extend((kmeans.label() + self.nCluster_per_block*i).astype(int));
					self.cCentro.extend(kmeans.centro());
			
				label_temp = np.array(label_temp, dtype=int)
				self.cCentro = np.array(self.cCentro)
				np.set_printoptions(threshold=10)
				if len(label_temp)!= self.XXnCluster_per_block*self.nBlock:
					print("ERROR len is {}, it must be {}".format(len(label_temp), self.nCluster_per_block*(int(self.nhWeight/self.nhWeight_per_block)+1)))
					print("{}-{}-{}".format(self.nCluster_per_block, self.nhWeight, self.nhWeight_per_block))
					return;


				for i in range(self.nhWeight):
					for i2 in range(self.nwWeight):
						self.cLabel[i][i2] = label_temp[XXarray[i][i2]];
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

		else :  #-------------------------------------------------------------------------
			self.nhWeight = cWeights.shape[0]
			self.nwWeight = cWeights.shape[1]
			self.nciWeight = cWeights.shape[2] #1
			self.ncoWeight = cWeights.shape[3] #32

			print("!!!!Tfc !!!",blocked , "------ ", blocked_param );
			print("--------", Conv_FC , "------------");
			if blocked :
				if cWeights.shape[3] < blocked_param:
					blocked_param = cWeights.shape[3];
				self.nCluster_per_block = int(nCluster/blocked_param);
				self.XXnCluster_per_block = int(XXnCluster/blocked_param);
				print("ncluster/block is", self.nCluster_per_block);
				print("XXncluster/block is", self.XXnCluster_per_block);

				if self.ncoWeight%blocked_param == 0 :
					self.ncoWeight_per_block = int(self.ncoWeight/blocked_param);
					self.nBlock = int(blocked_param)
				else :
					self.ncoWeight_per_block = int(self.ncoWeight/blocked_param)+1;
					self.nBlock = int(self.ncoWeight/self.ncoWeight_per_block)+1
				print("ncoWeight/block is", self.ncoWeight_per_block);

			# 3, 3, 1, 32
			print("cWeights.shape is {}, {}, {}, {}".format(cWeights.shape[0],cWeights.shape[1],cWeights.shape[2],cWeights.shape[3]))

			self.cWeights = cWeights
			self.cLabel = -np.ones((self.nhWeight, self.nwWeight, self.nciWeight, self.ncoWeight), dtype=int)		#nothing

			self.before_cCentro = -np.ones(XXnCluster)						#nothing
			for i in range(self.ncoWeight):
				for i2 in range(self.nciWeight):
					for i3 in range(self.nwWeight):
						for i4 in range(self.nhWeight):
							if self.before_cCentro[XXarray[i4][i3][i2][i]]==-1:
								self.before_cCentro[XXarray[i4][i3][i2][i]]=cWeights[i4][i3][i2][i];
							elif self.before_cCentro[XXarray[i4][i3][i2][i]]!=cWeights[i4][i3][i2][i]:
								print("ERROR!!!");
			if blocked :
				#STEP1 : hash index change
				label_temp = []
				self.cCentro = []
				for i in range(self.nBlock):
					kmeans = kmeans_cluster(\
						self.before_cCentro[i*self.XXnCluster_per_block:(i+1)*self.XXnCluster_per_block],\
						nCluster=self.nCluster_per_block,\
						max_iter=max_iter);
					label_temp.extend((kmeans.label() + self.nCluster_per_block*i).astype(int));
					self.cCentro.extend(kmeans.centro());
			
				label_temp = np.array(label_temp, dtype=int)
				self.cCentro = np.array(self.cCentro)
				np.set_printoptions(threshold=10)
				if len(label_temp)!= self.XXnCluster_per_block*self.nBlock:
					print("ERROR len is {}, it must be {}".format(len(label_temp), self.nCluster_per_block*(int(self.nhWeight/self.nhWeight_per_block)+1)))
					print("{}-{}-{}".format(self.nCluster_per_block, self.nhWeight, self.nhWeight_per_block))
					return;


				for i in range(self.ncoWeight):
					for i2 in range(self.nciWeight):
						for i3 in range(self.nwWeight):
							for i4 in range(self.nhWeight):
								self.cLabel[i4][i3][i2][i] = label_temp[XXarray[i4][i3][i2][i]];
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
				for i in range(self.ncoWeight):
					for i2 in range(self.nciWeight):
						for i3 in range(self.nwWeight):
							for i4 in range(self.nhWeight):
								self.cLabel[i4][i3][i2][i] = label_temp[XXarray[i4][i3][i2][i]];
				print("size of !!label is {}".format(self.cLabel)) 

				#STEP2 : get centroid by cLabel
				self.cCentro = kmeans.centro();

			print("before---------------------- is {}".format(self.before_cCentro)) 
			print("after----------------------- is {}".format(self.cCentro)) 
		

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
	def num_block(self):
		return self.nBlock

