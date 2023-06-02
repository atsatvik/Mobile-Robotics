import pandas as pd
import numpy as np
import plotly.graph_objs as go


def Estimatecorrespondences(X, Y, t, R, d_max):
    C = []
    x_corr = np.dot(X,R) + t 
    for i in range(len(X)):
        norm = np.linalg.norm(Y-x_corr[i], axis=1)
        y_corres = np.argmin(norm)
        if norm[y_corres]<d_max :
            C.append((i,y_corres))
    return np.array(C)



def ComputeoptimalRigidRegistration(X,Y,C):

    x_cent = np. mean(X[C[:,0]], axis=0)
    y_cent = np. mean(Y[C[:,1]], axis=0)

    x_devi = X[C[ :,0]] - x_cent
    y_devi = Y[C[:,1]] - y_cent

    
    w = np.dot(x_devi.T, y_devi)

    u, s, v = np.linalg.svd(w)

    Rot = np.dot(u, v)

    trans = y_cent - np.dot(x_cent, Rot)

    return Rot, trans



def ICP_alg(X , Y , to, Ro, d_max, max_iter):
    for i in range(max_iter):
        C = Estimatecorrespondences(X, Y, to, Ro, d_max)
        R, t = ComputeoptimalRigidRegistration(X,Y,C)
        to = t
        Ro = R
    return t, R, C


if __name__ == "__main__":

    dfx = pd.read_csv("pclX.txt" , sep=" ", header = None)
    X = dfx.to_numpy(dtype=float)

    dfy = pd.read_csv('pclY.txt', sep=" ", header = None)
    Y=dfy.to_numpy(dtype=float)

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(name= 'X', mode = 'markers', x=X[:,0], 
        y=X[:,1], z=X[:,2], marker=dict(color='rgb(256,0,0)', size=1)))
    fig.add_trace(go.Scatter3d(name= 'Y',mode = 'markers', x=Y[:,0], 
        y=Y[:,1], z=Y[:,2], marker=dict(color='rgb(0,0,256)', size=1)))
    fig.show()

    t = np.zeros((1,3))
    R = np. array([[1,0,0],[0,1,0], [0,0,1]])
    d_max = 0.25
    iter_val = 30
    t, R, C = ICP_alg(X, Y, t, R, d_max, iter_val)
    print("Rotation Matrix","\n", R, "\n")
    print("Translation Matrix","\n", t, "\n")

    X_corres = X[C[:,0]]
    Y_corres = Y[C[:,1]]
   

    corr_X = np.dot(X,R) + t
    err = Y[C[:,1]] - corr_X[C[:,0]]

    sq_err = np.square(np.linalg.norm(err,axis=1))
    MSE = sq_err.sum()/len(X)
    RMSE = np.sqrt(MSE)
    print("RMSE: ", RMSE)

    fig.add_trace(go.Scatter3d(name= 'X', mode = 'markers', x=X[:,0], 
        y=X[:,1], z=X[:,2], marker=dict(color='rgb(256,0,0)', size=1)))
    fig.add_trace(go.Scatter3d(name= 'X', mode = 'markers', x=Y[:,0], 
        y=Y[:,1], z=Y[:,2], marker=dict(color='rgb(0,0,256)', size=1)))
    fig.add_trace(go.Scatter3d(name= 'corrected X', mode = 'markers', x=corr_X[:,0], y=corr_X[:,1], 
        z=corr_X[:,2], marker=dict(color='rgb(0,256,0)', size=1)))
    fig.show()




