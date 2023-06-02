from PIL import Image
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt

def RecoverPath(s,g,pred):
	finallist = [g]
	current = g

	while current!=s:
		prev = pred[current]		 
		finallist.append(prev)
		current = prev
	return finallist

def N(v):
	row = v[0]
	col = v[1]
	nbors = [
			 (row-1,col-1),(row-1,col),(row-1,col+1),
			 (row,col-1),(row,col+1),
			 (row+1,col-1),(row+1,col),(row+1,col+1)
			]
	finallist = []
	for tup in nbors:
		if (occupancy_grid[tup[0],tup[1]] == 1):
			finallist.append(tup)
	return finallist

def d(v1,v2):
	return ((v1[1]-v2[1])**2 + (v1[0]-v2[0])**2)**(1/2)


def A_Star_Search(V, s, g, N, d):

	CostTo = {} 	
	pred = {}
	EstTotalCost = {}

	for row in range(len(V)):
		for col in range(len(V[0])):
			CostTo.update({(row,col):100000})
			EstTotalCost.update({(row,col):100000})

	CostTo.update({s:0})
	EstTotalCost.update({s:d(s,g)})

	Q = [(d(s,g),s)]

	while len(Q)>0:
		Q = sorted(Q)
		v = Q[0][1]
		Q = Q[1:len(Q)]

		if v == g:
			return RecoverPath(s,g,pred)

		for i in N(v):
			pvi = CostTo[v] + d(v,i)

			if pvi < CostTo[i]:
				pred.update({i:v})
				CostTo.update({i:pvi})
				EstTotalCost.update({i: pvi + d(i,g)})

				check = False
				for n in range(len(Q)):
					if Q[n][1] == i:
						check = True
						idx = n

				if check:
					Q[idx] = (EstTotalCost[i],Q[idx][1])
				else:
					Q.append((EstTotalCost[i],i))

	return []


if __name__ == '__main__':

	# Read image from disk using PIL
	path  = 'occupancy_map.png'

	occupancy_map_img = Image.open(path)

	# Interpret this image as a numpy array, and threshold its values to→ {0,1}
	occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)
	opt_path = A_Star_Search(occupancy_grid,(635, 140),(350, 400),N,d)

	data = image.imread(path)
	xs = [points[1] for points in opt_path]
	ys = [points[0] for points in opt_path]
	plt.plot(xs, ys)
	plt.imshow(data)
	plt.show()

	






