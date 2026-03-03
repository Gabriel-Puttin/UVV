from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

iris_dataset = load_iris()
iris_dataframe = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)

iris_dataframe['Iris type'] = iris_dataset['target']
iris_dataframe['Species'] = iris_dataframe['Iris type'].apply(lambda x: 'setosa' if
x == 0 else ('versicolor' if x == 1 else 'virginica'))

versicolor = iris_dataframe.loc[iris_dataframe['Species'] == "versicolor"]
setosa = iris_dataframe.loc[iris_dataframe['Species'] == "setosa"]
virginica = iris_dataframe.loc[iris_dataframe['Species'] == "virginica"]

def scatter_1d():
    fig, axs = plt.subplots(2, 2, figsize=(20, 12), facecolor='w', edgecolor='k')
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.06, wspace = 0.3,
    hspace = 0.3)
    a = 0
    for i in range(2):
        for j in range(2):
            axa = axs[i, j]
            axa.set_title("Univariate of {}".format(iris_dataframe.columns[a]),
            fontsize = '20')

            axa.plot(versicolor[iris_dataframe.columns[a]], np.zeros_like(versicolor
            [iris_dataframe.columns[a]]), 'o', color='b', markersize=15, alpha
            =0.6, label = 'Versicolor')
            axa.plot(setosa[iris_dataframe.columns[a]], np.zeros_like(setosa[
            iris_dataframe.columns[a]]), 'o', color='r', markersize=11, alpha =0.6, label = 'Setosa')
            axa.plot(virginica[iris_dataframe.columns[a]], np.zeros_like(virginica[
            iris_dataframe.columns[a]]), 'o', color='g', markersize=8, alpha
            =0.8, label = 'Virginica')

            axa.set_yticks([])
            axa.tick_params(axis='x', labelsize=20)
            axa.xaxis.label.set_fontsize(15)
            axa.set_xlabel("Values")

            a += 1

        axs[0, 0].text( 3.8, 0.045, 'a)',fontsize=30)
        axs[0, 1].text( 1.7, 0.045, 'b)',fontsize=30)
        axs[1, 0].text(0.25, 0.045, 'c)',fontsize=30)
        axs[1, 1].text(-0.2, 0.045, 'd)',fontsize=30)
        plt.legend()
        plt.savefig("scatter_1d_sepal_lenght_01.png", dpi=300)

if __name__ == '__main__':
    scatter_1d()
