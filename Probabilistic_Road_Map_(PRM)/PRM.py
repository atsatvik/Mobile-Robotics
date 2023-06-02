from PIL import Image
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import networkx as nx


def d(v1,v2):
	return ((v1[1]-v2[1])**2 + (v1[0]-v2[0])**2)**(1/2)

def check_reach(occupancy_grid, v, vnew, d_max):
	x1 = v[1] 
	y1 = v[0] 
	x2 = vnew[1] 
	y2 = vnew[0]

	if x1-x2!=0:
		m = (y2-y1)/(x2-x1)
		metric = (2**0.5)/2
		c = y1 - m*x1
	else:
		metric = 0.5
	
	nearby_vox = []

	if vnew[1]<v[1]:
		#right of vnew and above
		if vnew[0]<v[0]:
			for i in range(abs(v[0]-vnew[0])+1):
				for j in range(abs(v[1]-vnew[1])+1):
					nearby_vox.append((vnew[0]+i,vnew[1]+j))
		#right of vnew and below			
		if vnew[0]>v[0]:
			for i in range(abs(v[0]-vnew[0])+1):
				for j in range(abs(v[1]-vnew[1])+1):
					nearby_vox.append((vnew[0]-i,vnew[1]+j))
	
	if vnew[1]>v[1]:
		#left of vnew and above					
		if vnew[0]<v[0]:
			for i in range(abs(v[0]-vnew[0])+1):
				for j in range(abs(v[1]-vnew[1])+1):
					nearby_vox.append((vnew[0]+i,vnew[1]-j))
		#left of vnew and below			
		if vnew[0]>v[0]:
			for i in range(abs(v[0]-vnew[0])+1):
				for j in range(abs(v[1]-vnew[1])+1):
					nearby_vox.append((vnew[0]-i,vnew[1]-j))
	
	#same vertical line and below vnew
	if vnew[1] == v[1] and vnew[0]>v[0]:
		for i in range(abs(vnew[0]-v[0])+1):
			nearby_vox.append((vnew[0]-i,vnew[1]))

	#same vertical line and above vnew
	if vnew[1] == v[1] and vnew[0]<v[0]:
		for i in range(abs(vnew[0]-v[0])+1):
			nearby_vox.append((vnew[0]+i,vnew[1]))

	#same horizontal line and left of vnew
	if vnew[0] == v[0] and vnew[1]>v[1]:
		metric = 0.5
		for i in range(abs(vnew[1]-v[1])+1):
			nearby_vox.append((vnew[0],vnew[1]-i))

	#same horizontal line and right of vnew
	if vnew[0] == v[0] and vnew[1]<v[1]:
		metric = 0.5
		for i in range(abs(vnew[1]-v[1])+1):
			nearby_vox.append((vnew[0],vnew[1]+i))


	for vert in nearby_vox:
		x = vert[1]
		y = vert[0]
		
		if metric == 0.5:
			if occupancy_grid[vert[0]][vert[1]] == 0:
				return False
		else:
			dist = (abs(y - m*x - c))/(1+m**2)**0.5
			if dist<(2**0.5)/2:
				if occupancy_grid[vert[0]][vert[1]] == 0:
					return False
	return True
	

def AddVertex(G, vnew, d_max, occupancy_grid):
	G.add_node(vnew)
	for v in G:
		if v!=vnew and  d(v,vnew)<=d_max:
			if check_reach(occupancy_grid, v, vnew, d_max):
				G.add_edge(v,vnew,weight=d(v,vnew))


def get_point(occupancy_grid):

	while True:
		row = np.random.randint(0,len(occupancy_grid),1)
		col = np.random.randint(0,len(occupancy_grid[0]),1)
		if occupancy_grid[row[0]][col[0]] == 1:
			return (row[0],col[0])


def ConstructPRM(N, d_max, occupancy_grid):
	G = nx.Graph()
	for i in range(N):
		if i==0:
			vnew = (635, 140)
		if i==1:
			vnew = (350, 400)
		if i>1:
			vnew = get_point(occupancy_grid)
		AddVertex(G, vnew, d_max, occupancy_grid)
	return G


if __name__ == '__main__':

	path  = 'occupancy_map.png'
	occupancy_map_img = Image.open(path)
	occupancy_grid = (np.asarray(occupancy_map_img) > 0).astype(int)
	N = 500
	d_max = 75
	G = ConstructPRM(N,d_max,occupancy_grid)

	data = image.imread(path)

	points = G.edges
	fig = plt.figure(1)

	for points in G.edges:

		p1x = points[0][1]
		p1y = points[0][0]
		p2x = points[1][1]
		p2y = points[1][0]

		xs = [p1x,p2x]
		ys = [p1y,p2y]
		plt.plot(xs,ys,color='black')
		
	s = (635, 140) 
	g = (350, 400)
	
	opt_path = nx.astar_path(G, s, g, heuristic=d, weight="weight")

	length_of_path = 0
	for i in range(len(opt_path)-1):
		length_of_path += d(opt_path[i],opt_path[i+1])

	print(opt_path[0],opt_path[-1])
	
	xs = [point[1] for point in opt_path]
	ys = [point[0] for point in opt_path]
	plt.plot(xs,ys, color='green')
	plt.imshow(data)
	plt.show()

