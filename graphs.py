from numpy import tile
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
#import numpy as np
sns.set_theme()

#####################################
######## Main Program Driver ########
#####################################

if __name__ == "__main__":
    #Import the information
    df_d = {}
    for i in range(1,3):
        df_d[i] = pd.read_csv ("./data/example"+str(i)+"_classified.csv", header=0) 
        df_d[i]['Cluster'] = df_d[i]['Cluster'].astype(str)

    #Elaborate and the plot
    sns.set(font_scale = 2)
    fig, ax =plt.subplots(2,2, figsize=(20,20))
    sns.scatterplot(data = df_d[1], x="X1", y="X2", legend=False, ax=ax[0][0]).set(title='Example 1: Input Data')
    sns.scatterplot(data = df_d[1], x="X1", y="X2", hue="Cluster", legend=False, ax=ax[0][1]).set(title='Example 1: Classified')
    sns.scatterplot(data = df_d[2], x="X1", y="X2", legend=False, ax=ax[1][0]).set(title='Example 2: Input Data')
    sns.scatterplot(data = df_d[2], x="X1", y="X2", hue="Cluster", legend=False, ax=ax[1][1]).set(title='Example 2: Classified')
    #fig.show()
    plt.savefig('./graph/em_clustering.png')