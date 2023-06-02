import numpy as np
import scipy.linalg
import scipy.stats
import matplotlib.pyplot as plt
import random


def ParticleFilterPropagate(X_t_1,t_1,phi_l,phi_r,t_2,r,w,sigma_l,sigma_r):
    X_t_2=np.zeros((X_t_1.shape[0],X_t_1.shape[1],X_t_1.shape[2]))
    dt=t_2-t_1
    for i in range(0,len(X_t_1)):
        err_l=sigma_l*np.random.randn()
        err_r=sigma_r*np.random.randn()
        phi_l_true=phi_l+err_l
        phi_r_true=phi_r+err_r
        omega_true=np.array([[0, -r/w*(phi_r_true-phi_l_true), r/2*(phi_r_true+phi_l_true)],[r/w*
        (phi_r_true-phi_l_true), 0, 0],[0,0,0]])
        X_t_2[i]=np.matmul(X_t_1[i],scipy.linalg.expm(dt*omega_true))
    return X_t_2


def ParticleFilterUpdate(z_t, x_t, sigma_p):
    w = np.zeros((len(x_t),1))
    for i in range(len(x_t)):
        w[i] = scipy.stats.multivariate_normal.pdf(z_t, mean = [x_t[i,0,2], x_t[i,1,2]] , 
            cov = (sigma_p**2)*np.array([[1,0],[0,1]]))
        bel_X = np.empty((x_t.shape[0], x_t.shape[1], x_t.shape[2]))
        bel_X = np.array(random.choices(x_t, weights= w, k = 1000))
    return bel_X


def Extracredit_e(N=1000):
    X_0 = np.array([[1,0,0],[0,1,0],[0,0,1]])
    X_0 = np.tile(X_0, (N,1,1))
    t_2 = 10
    phi_l = 1.5
    phi_r= 2
    r= 0.25
    w= 0.5
    sigma_l = 0.05
    sigma_r = 0.05
    t_1 = 0
    X_2 = ParticleFilterPropagate(X_0,t_1,phi_l,phi_r,t_2,r,w,sigma_l,sigma_r)
    mean=np.mean(X_2[:,:-1,2], axis = 0)
    covar=np.cov(X_2[:,:-1,2], rowvar = False)
    plt.scatter(X_2[:,0,2], X_2[:,1,2],marker="^",s=1,c="green",label="At t=10")
    plt.scatter(0,0,marker="*",s=100,c='red',label="At t=0")
    plt.xlabel("Translation in X direction at t=10")
    plt.ylabel("Translation in Y direction at t=10")
    plt.legend()
    plt.grid()
    plt.show()
    print("Mean of translation of X and Y particles at t=10","\n")
    print(mean,"\n")
    print("Covariance of translation of X and Y particles  at t=10","\n")
    print(covar)



def Extracredit_f(N=1000):
    X = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
    X = np.tile(X, (N,1,1))
    t_2 = [5,10,15,20]
    phi_l = 1.5
    phi_r = 2
    r = 0.25
    w = 0.5
    sigma_l = 0.05
    sigma_r = 0.05
    t_1 = 0
    mean = []
    covar = []

    for i in range(0,len(t_2)):
        X= ParticleFilterPropagate(X, t_1, phi_l, phi_r, t_2[i], r, w,sigma_l,sigma_r)
        mean.append(np.mean(X[:,:-1,2], axis = 0))
        covar.append(np.cov(X[:,:-1,2], rowvar = False))
        t_1 = t_2[i]
        plt.scatter(X[:,0,2], X[:,1,2], s=1,marker="^", label="location at time: %i" %(t_1))
    plt.scatter(0,0,marker="*",s=100,c='red',label="Start Point")
    plt.legend()
    plt.xlabel("Translation in X")
    plt.ylabel("Translation in Y")
    plt.grid()
    plt.show()

    for i in range(0,len(mean)):
        print("Mean at time %i is: %s "%(t_2[i], mean[i]),"\n")
    for i in range(len(covar)):
        print("Covariance at time %i is: %s"%(t_2[i],covar[i]))


def Extracredit_g(N=1000):
    X = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
    X = np.tile(X, (N,1,1))
    t_2 = [5,10,15,20]
    phi_l = 1.5
    phi_r = 2
    r = 0.25
    w = 0.5
    sigma_l = 0.05
    sigma_r = 0.05
    sigmap = 0.10
    t_1 = 0
    mean = []
    covar = []
    obs = np.asarray([[1.6561, 1.2847], [1.0505, 3.1059], 
        [-0.9875, 3.2118], [-1.6450, 1.1978]])
    for i in range(len(t_2)):
        X = ParticleFilterPropagate(X,t_1,phi_l,phi_r,t_2[i],r,w,sigma_l,sigma_r)
        X = ParticleFilterUpdate(obs[i],X , sigmap)
        mean.append(np.mean(X[:,:-1,2], axis = 0))
        covar.append(np.cov(X[:,:-1,2], rowvar = False))
        t_1=t_2[i]
        plt.scatter(X[:,0,2], X[:,1,2], s=1,marker='x', label="Belief of particle at time: %i"%(t_1))
    plt.scatter(0,0,c="red",marker='*',label='Starting Point')
    plt.legend()
    plt.xlabel("Translation in X")
    plt.ylabel("Translation in Y")
    plt.grid()
    plt.show()

    for i in range(0,len(mean)):
        print("Mean at time %i is: %s "%(t_2[i], mean[i]),"\n")
    for i in range(len(covar)):
        print("Covariance at time %i is: %s"%(t_2[i],covar[i]))


if __name__ == "__main__":

    Extracredit_e()
    Extracredit_f()
    Extracredit_g()