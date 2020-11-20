import numpy as np
    
#####################################
######### Helper  Functions #########
#####################################

#-----------------------------------
def write_examples(data, i):
    """ Write results
    Args:
        data (numpy array): n + 1 by d + 1 array
        i: integer which corresponds to the example number
    """
    #Save csv file
    np.savetxt("./data/example"+str(i)+".csv", data[1:,1:], delimiter=',', comments="")

#####################################
######## Main Program Driver ########
#####################################

if __name__ == "__main__":
    #Define number of observations for each cluster and number of examples generated
    n = 300
    examples = 4
    np.random.seed(1)

    for i in range(4):
        #Separate clusters
        apart = i + 0.01*(1+i)
        means = np.array([[1,1+apart],[1,1+apart]])
        sigmas = np.array([[0.7,0.7],[0.7,0.7]])
        (dimensions, clusters) = means.shape

        #Create data and write
        data = np.zeros(dimensions + 1)
        for cluster in range(clusters):
            instances = np.array([cluster + 1 for i in range(n)])
            for dimension in range(dimensions):
                column = np.random.normal(means.T[cluster][dimension], sigmas.T[cluster][dimension], n)
                instances = np.vstack((instances, column))
            data = np.vstack((data, instances.T))
            write_examples(data,i)