from PIL import Image
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import networkx as nx
from skimage.draw import line

# Get coordinates, r=rows, c=cols of your line




def d(v1,v2):
	return ((v1[1]-v2[1])**2 + (v1[0]-v2[0])**2)**(1/2)

def check_reach(occupancy_grid, v, vnew, d_max):
	x1 = v[1] 
	y1 = v[0] 
	x2 = vnew[1] 
	y2 = vnew[0] 

	rr, cc = line(x1,y1,x2,y2)
	points = list(zip(rr,cc))
	
	for vert in points:
		if occupancy_grid[vert[1]][vert[0]] == 0:
			return False
	return True
	

def AddVertex(G, vnew, d_max, occupancy_grid):
	G.add_node(vnew)
	for v in G:
		if v!=vnew and  d(v,vnew)<=d_max:
			if check_reach(occupancy_grid, v, vnew, d_max):
				G.add_edge(v,vnew,weight=d(v,vnew))

	# return G

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

	path  = 'C:/Users/91991/OneDrive/Desktop/Mobile_HW2/occupancy_map.png'
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
	xs = [point[1] for point in opt_path]
	ys = [point[0] for point in opt_path]
	plt.plot(xs,ys,color='green')
	plt.imshow(data)
	plt.show()

