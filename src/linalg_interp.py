"""Functions for GOPH 419 Lab Assignment #1."""
import numpy as np
#water_data = load_data('water_density_vs_temp_usgs.txt')

#air_data = load_data('air_density_vs_temp_eng_toolbox.txt')

def gauss_iter_solve(A, b, x0, tol= 1e-8, alg ="seidel", max_iter=50):
    k =0 
    eps_a=1
    x = np.array(x0, dtype = float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if str.strip(alg).lower() == "seidel":

        while eps_a >tol and k< max_iter:
            x_old = np.array(x)
            for i in range(len(x)):
                x[i] = (b[i]-A[i,:i]@ x[:i]-A[i,i+1:]@x[i+1:])/A[i,i]
                
                
            dx= x-x_old
            eps_a = np.linalg.norm(dx)/np.linalg.norm(x)
            k+=1
  

    elif str.strip(alg).lower() == "jacobi":


        while eps_a >tol and k< max_iter:
            x_old = np.array(x)
            for i in range(len(x)):
                x[i] = (b[i]-A[i,:i]@ x_old[:i]-A[i,i+1:]@x_old[i+1:])/A[i,i]
                
                
            dx= x-x_old
            eps_a = np.linalg.norm(dx)/np.linalg.norm(x)
            k+=1
    return x

def spline_function(xd, yd, order=3):
    dy = np.diff(yd)
    dx = np.diff(xd)
    F = dy / dx
    N= len(F)

    if order == 1:
        def s1(x):
            k = np.maximum(np.searchsorted(xd, x) - 1, 0)
            y = yd[k] + F[k] * (x - xd[k])
            return y

        return s1

    elif order == 2:
        def s2(x):
            k = np.maximum(np.searchsorted(xd, x) - 1, 0)
            if k>=len(F):
                k=len(F)-1
            ak = yd[k]
            #b = F[k]
            #c = (F[k + 1] - F[k]) / (xd[k + 1] - xd[k]) - 2 * F[k] / (xd[k + 1] - xd[k])
            A0= np.hstack([np.diag(dx[:-1]),np.zeros((N-1,1))])
            A1= np.hstack([np.zeros((N-1,1)),np.diag(dx[1:])])
            A2 =np.zeros((1,N))
            A2[0,0]=1.0
            A2[0,1]=-1.0
            A=np.vstack([A2,A0+A1])
            B=np.zeros((N,))
            B[1:]= np.diff(F)
            c= np.linalg.solve(A,B)
            b=F-c*dx
            bk=b[k]
            ck=c[k]

            y = ak + bk * (x - xd[k]) + ck * (x - xd[k]) ** 2
            return y

        return s2

    elif order == 3:
        def s3(x):
            k = np.maximum(np.searchsorted(xd, x) - 1, 0)
            if k>=len(F):
                k=len(F)-1
            A =np.zeros((N+1,N+1))
            A[1:-1,1:-1]+=np.diag(2*(dx[:-1]+dx[1:]))
            A[1:-1,:-2]+= np.diag(dx[:-1])
            A[1:-1,2:]+=np.diag(dx[1:])
            A[0,:3]+=[-dx[1],dx[0]+dx[1],-dx[0]]
            A[-1,-3:]+=[-dx[-1],dx[-1]+dx[-2],-dx[-2]]
            B=np.zeros((N+1,))
            B[1:-1]= 3*np.diff(F)
            c= np.linalg.solve(A,B)
            d=np.diff(c)/(3*dx)
            b=F-c[:-1]*dx-d*dx**2
            bk=b[k]
            ck=c[k]
            dk=d[k]
            ak = yd[k]
            
            y = ak + bk * (x - xd[k]) + ck * (x - xd[k]) ** 2 + dk * (x - xd[k]) ** 3
            return y

        return s3

    else:
        raise ValueError("Order must be 1, 2, or 3.")
 

           
       





