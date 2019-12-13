import numpy as np

#a = np.random.randn(10,10)
#print("this is ", a)

#a -= 1
#print("this is ", a)

#a = abs(a)
#print("this is ", a)



#a = np.array([-2, 1, -1, 3, -4, 5, 6, 7, 8, 9])
#print("this is ", a)
#print("this is ", np.argmin(a))


#a = np.absolute(a)
#print("this is ", a)

#b =np.array([2, 1, 1, 1, 1, 1, 1, 1, 1, 1])
#a = a/ b
#print("this is ", a)
#print("this is ", a.astype(np.int))

#c = 1
#a = c - a
#print("this is ", a)

'''
a1 = np.array([[0.1, 0.2, 0.3], 
			[0.4, 0.5, 0.6], 
			[0.7, 0.8, 0.9]])

a = np.array([[2, 0, 0], 
			[3, 1, 2], 
			[1, 1, 1]])
b = np.array([1.2, 2.3, 3.4, 4.5])
c = np.zeros(4, dtype=float)

print(b[a])
'''

'''
index = np.zeros((3,3, 2), dtype=int)
for i in range(3):
	for i2 in range(3):
		index[i][i2] = [i, i2]
		
print(index[2][1][0])
'''
X = 0x00000000;
Y = 0x00000001;
Z = (Y<<32) + X;
print("X {}, Y {}, Z {}".format(X,Y,Z));
print("X {}, Y {}, Z {}".format(X.bit_length(),Y.bit_length(),Z.bit_length()));

def newHash(i, i2):
	
	#initialize
	centroid = ((i & 0xFFFF)<<16)+(i2 & 0xFFFF)
	#temp = (((i & 0xFFFF)*0x8521)+((i2 & 0xFFFF)*0x1252) + 0x3424 ) ^ 0x4920;
	
	#case1	
	#print("real value temp is ", bin(temp))
	centroid ^= (centroid <<13) & 0xFFFFFFFF;
	centroid ^= (centroid >>7) & 0xFFFFFFFF;
	centroid ^= (centroid <<5) & 0xFFFFFFFF;
	centroid = (centroid * 0x2545F4914F6CDCD1D) & 0xFFFFFFFFFFFFFFFF;
	#print("length is {} {} = {} ".format(i.bit_length(), i2.bit_length(), temp.bit_length()))
	

	#case2
	#centroid = i * 3266489917 + 374761393;
	#centroid = (centroid<< 17) | (centroid>> 15);

	#centroid += i2 * 3266489917;
	
	#centroid *= 668265263;
	#centroid ^= centroid > 15;
	#centroid *= 2246822519;
	#centroid ^= centroid > 13;
	#centroid *= 3266489917;
	#centroid ^= centroid > 16;
	
	return centroid%128

'''
list1=np.zeros((10,10),dtype=int)
list2=np.zeros(128, dtype=int)
for i in range(10):
	print("-------------");
	for i2 in range(10):
		list1[i][i2] = newHash(i,i2);
		list2[list1[i][i2]] += 1;
		#print("----------({},{}) is {}".format(i, i2, list1[i][i2]));
		print("({},{}) = {}".format(i, i2, list1[i][i2]), end=' ');
	print(" ");

print("==============let's see=================");
print(list2);
'''

def newHash2(i, i2, i3, i4):
	temp = ((i & 0xFFFF)<<48) + ((i2 & 0xFFFF)<<32) + ((i3 & 0xFFFF)<<16) + (i4 & 0xFFF)
	print("real value temp is ", bin(temp))
	temp ^= (temp <<13) & 0xFFFFFFFFFFFFFFFF;
	temp ^= (temp >>7) & 0xFFFFFFFFFFFFFFFF;
	temp ^= (temp <<17) & 0xFFFFFFFFFFFFFFFF;
	#temp = (temp * 0x2545F4914F6CDCD1D) & 0xFFFFFFFFFFFFFFFF;
	print("length is {} {} {} {} = {}".format(i.bit_length(),i2.bit_length(),i3.bit_length(),i4.bit_length(), temp.bit_length()))
	return temp%256
'''
list3=np.zeros((5,5,5,5), dtype=int)
list4=np.zeros(256, dtype=int)
for i in range(4):
	for i2 in range(4):
		for i3 in range(4):
			for i4 in range(4):
				list3[i][i2][i3][i4] = newHash2(i,i2,i3,i4);
				list4[list3[i][i2][i3][i4]] += 1;
				print("-----------({},{},{},{}) is {}".format(i, i2, i3, i4, list3[i][i2][i3][i4]));


print("==============let's see=================");
print(list4);

'''
def XXhash_real(x, y, nCluster=128, SEED=0):
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
def XXhash_real2(i, i2, i3, i4, nCluster=128, SEED=0):
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
'''
list5=np.zeros((10,10),dtype=int)
list6=np.zeros(128, dtype=int)
for i in range(10):
	print("-------------");
	for i2 in range(10):
		for i3 in range(10):
			for i4 in range(10):
				list5[i][i2] = XXhash_real(i,i2);
				list6[list5[i][i2]] += 1;
				print("({},{}) = {}".format(i, i2, list5[i][i2]), end=' ');
				#print(list5[i][i2], end='', flush=True);
				#print("hello", end='')
		print(" ");

print("==============let's see=================");
print(list6);
'''

list5=np.zeros((10,10,10,10),dtype=int)
nCentroid = 10000
list6=np.zeros(nCentroid, dtype=int)
for i in range(10):
	#print("---------------------------------");
	for i2 in range(10):
		#print("-------------");
		for i3 in range(10):
			for i4 in range(10):
				list5[i4][i3][i2][i] = XXhash_real2(i,i2,i3,i4, nCluster=nCentroid);
				list6[list5[i4][i3][i2][i]] += 1;
				#print("({},{}) = {}".format(i, i2, list5[i4][i3][i2][i]), end=' ');
		#print(" ");
	#print(" ");
import sys
np.set_printoptions(threshold=sys.maxsize)
print("==============let's see=================");
print(list6);
print("0 is ", nCentroid - np.count_nonzero(list6))
print("0 percentange is ", (nCentroid - np.count_nonzero(list6))/nCentroid)
print("nonzero is ", np.count_nonzero(list6))
print("1 is ", np.count_nonzero(list6 == 1))
print("1 percentange is ", (np.count_nonzero(list6==1))/nCentroid)
print("2 is ", np.count_nonzero(list6 == 2))
print("2 percentange is ", (np.count_nonzero(list6==2))/nCentroid)
print("3 is ", np.count_nonzero(list6 == 3))
print("3 percentange is ", (np.count_nonzero(list6==3))/nCentroid)
print("4 is ", np.count_nonzero(list6 == 4))
print("4 percentange is ", (np.count_nonzero(list6==4))/nCentroid)
print("5 is ", np.count_nonzero(list6 == 5))
print("5 percentange is ", (np.count_nonzero(list6==5))/nCentroid)
print("6 is ", np.count_nonzero(list6 == 6))
print("6 percentange is ", (np.count_nonzero(list6==6))/nCentroid)

print("list5 shape is ", list5.shape)
print("list5 shape[0] is ", list5.shape[0])
print("list5[0][0][0][1] is ", list5[0][0][0][1])
print("list5[0][0][0][2] is ", list5[0][0][0][2])
