import numpy as np
    
#####################################
######## Main Program Driver ########
#####################################

if __name__ == "__main__":
    #Define number of observations and variance of clusters
    n = 300
    var = 0.7
    np.random.seed(1)

    means = [[2*[distance*i] for i in range(2)] for distance in (1, 2)] 
    arrays = [[np.random.multivariate_normal(mean=m, cov=var*np.identity(2), size=n).T 
                for m in values] for values in means]
    arrays = [np.concatenate(array, axis=1) for array in arrays]

    for i in range(len(arrays)):
        np.savetxt("./data/example"+str(i+1)+".csv", arrays[i].T, delimiter=',', comments="")