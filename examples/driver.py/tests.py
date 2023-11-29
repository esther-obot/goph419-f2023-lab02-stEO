"""Tests for GOPH 419 Lab Assignment #2."""
import numpy as np
import matplotlib.pyplot as plt
from lab02.linalg_interp import gauss_iter_solve, spline_function

def test_gauss_iter_solve():
    A = np.array([[3,-1,0,0],
                  [-1,5,-1,0],
                  [0,-1,5,-1],
                  [0,0,-1,3]])
    
    B = np.array([1,2,3,4])
    X_np =np.linalg.solve(A,B)
    X =  gauss_iter_solve(A,B, np.zeros_like(B))
    X_jacobi =  gauss_iter_solve(A,B, np.zeros_like(B),alg = "jacobi")
    print(X)
    print(X_jacobi)
    print(X_np)
    

def test_linear_spline():
    # Unit tests for spline_function
    xd = np.linspace(1.0,4.0,4)
    yd = 0.5 +1.2*xd
    xp = np.linspace(1.0,4.0,20)
    s1 =spline_function(xd,yd,1)
    yp = np.array([s1(x)for x in xp])
    yp_exp =0.5+1.2*xp
    dy = yp_exp-yp
    eps_t = np.linalg.norm(dy)/np.linalg.norm(yp_exp)

    plt.figure()
    plt.plot(xd,yd,'xr',label="data")
    plt.plot(xp,yp,'ok',label="test")
    plt.text(2.0,2.0,f"eps_t ={eps_t}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("test_linear_spline.png")

def test_parabolic_spline():
    xd = np.linspace(1.0,4.0,4)
    yd = 0.5 + 1.2*xd + 0.5*xd**2
    xp = np.linspace(1.0,4.0,20)
    s2 =spline_function(xd,yd,2)
    yp = np.array([s2(x)for x in xp])
    yp_exp =0.5+1.2*xp + 0.5*xp**2
    dy = yp_exp-yp
    eps_t = np.linalg.norm(dy)/np.linalg.norm(yp_exp)

    plt.figure()
    plt.plot(xd,yd,'xr',label="data")
    plt.plot(xp,yp,'ok',label="test")
    plt.text(2.0,2.0,f"eps_t ={eps_t}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("test_parabolic_spline.png")

def test_cubic_spline():
    xd = np.linspace(1.0,4.0,4)
    yd = 0.5 + 1.2*xd + 0.5*xd**2+0.1*xd**3
    xp = np.linspace(1.0,4.0,20)
    s3 =spline_function(xd,yd,3)
    yp = np.array([s3(x)for x in xp])
    yp_exp =0.5+1.2*xp + 0.5*xp**2+0.1*xp**3
    dy = yp_exp-yp
    eps_t = np.linalg.norm(dy)/np.linalg.norm(yp_exp)

    plt.figure()
    plt.plot(xd,yd,'xr',label="data")
    plt.plot(xp,yp,'ok',label="test")
    plt.text(2.0,2.0,f"eps_t ={eps_t}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("test_cubic_spline.png")



# Run tests
if __name__ == "__main__":
    test_gauss_iter_solve()
    test_linear_spline()
    test_parabolic_spline()
    test_cubic_spline()








