
import numpy as np
import matplotlib.pyplot as plt
from lab02.linalg_interp import gauss_iter_solve
from lab02.linalg_interp import spline_function

#water_data = load_data('water_density_vs_temp_usgs.txt')
#air_data = load_data('air_density_vs_temp_eng_toolbox.txt')


def load_data(file_path):
    data=np.loadtxt(file_path) 
    xd = data[:,0]
    yd= data[:,1]

    return xd,yd

   

def main():
    T_air_data,rho_air_data = load_data('air_density_vs_temp_eng_toolbox.txt')
    T_water_data,rho_water_data = load_data('water_density_vs_temp_usgs.txt')
    s1_air= spline_function(T_air_data,rho_air_data,1)
    s1_water= spline_function(T_water_data,rho_water_data,1)
    s2_air= spline_function(T_air_data,rho_air_data,2)
    s2_water= spline_function(T_water_data,rho_water_data,2)
    s3_air= spline_function(T_air_data,rho_air_data,3)
    s3_water= spline_function(T_water_data,rho_water_data,3)
    T_air_plot = np.linspace(np.min(T_air_data),np.max(T_air_data),100)
    T_water_plot = np.linspace(np.min(T_water_data),np.max(T_water_data),100)
    rho_air_1= np.array([s1_air(T) for T in T_air_plot])
    rho_water_1= np.array([s1_water(T) for T in T_water_plot])
    rho_air_2= np.array([s2_air(T) for T in T_air_plot])
    rho_water_2= np.array([s2_water(T) for T in T_water_plot])
    rho_air_3= np.array([s3_air(T) for T in T_air_plot])
    rho_water_3= np.array([s3_water(T) for T in T_water_plot])
    plt.figure()
    plt.subplot(3,2,1)
    plt.plot(T_air_data,rho_air_data,'xr',label="data")
    plt.plot(T_air_plot,rho_air_1,'--k',label="linear")
    plt.ylabel("density")
    plt.title("air")
    plt.legend()

    plt.subplot(3,2,3)
    plt.plot(T_air_data,rho_air_data,'xr',label="data")
    plt.plot(T_air_plot,rho_air_2,'--k',label="quadratic")
    plt.ylabel("density")
    plt.legend()

    plt.subplot(3,2,5)
    plt.plot(T_air_data,rho_air_data,'xr',label="data")
    plt.plot(T_air_plot,rho_air_3,'--k',label="cubic")
    plt.ylabel("density")
    plt.legend()
    plt.xlabel("temperature")

    plt.subplot(3,2,2)
    plt.plot(T_water_data,rho_water_data,'xr',label="data")
    plt.plot(T_water_plot,rho_water_1,'--k',label="linear")
    plt.ylabel("density")
    plt.title("water")
    plt.legend()

    plt.subplot(3,2,4)
    plt.plot(T_water_data,rho_water_data,'xr',label="data")
    plt.plot(T_water_plot,rho_water_2,'--k',label="quadratic")
    plt.ylabel("density")
    plt.legend()

    plt.subplot(3,2,6)
    plt.plot(T_water_data,rho_water_data,'xr',label="data")
    plt.plot(T_water_plot,rho_water_3,'--k',label="cubic")
    plt.ylabel("density")
    plt.legend()
    plt.xlabel("temperature")
    
    plt.savefig("water_air_interp.png")




if __name__ == '__main__':
    main()