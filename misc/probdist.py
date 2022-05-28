import numpy

for j in range(1, 8):
	freq = [0 for x in range(10)]
	for i in range(10**j):
		freq[numpy.random.randint(10)]+=1
	print(freq)
