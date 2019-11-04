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


a1 = np.array([[0.1, 0.2, 0.3], 
			[0.4, 0.5, 0.6], 
			[0.7, 0.8, 0.9]])

a = np.array([[2, 0, 0], 
			[3, 1, 2], 
			[1, 1, 1]])
b = np.array([1.2, 2.3, 3.4, 4.5])
c = np.zeros(4, dtype=float)

print(b[a])
print(c[a] = a1)


'''
index = np.zeros((3,3, 2), dtype=int)
for i in range(3):
	for i2 in range(3):
		index[i][i2] = [i, i2]
		
print(index[2][1][0])
'''
