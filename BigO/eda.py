from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


def scatter_2d():
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), facecolor='w', edgecolor='k')
    plt.subplots_adjust(
        left=0.08, right=0.95, top=0.95, bottom=0.15,
        wspace=0.3, hspace=0.3
    )

    axs[0].scatter(
        np.array(versicolor[iris_dataframe.columns[0]]),
        np.array(versicolor[iris_dataframe.columns[1]]),
        c='b'
    )
    axs[0].scatter(
        np.array(setosa[iris_dataframe.columns[0]]),
        np.array(setosa[iris_dataframe.columns[1]]),
        c='r'
    )
    axs[0].scatter(
        np.array(virginica[iris_dataframe.columns[0]]),
        np.array(virginica[iris_dataframe.columns[1]]),
        c='g'
    )

    axs[0].yaxis.label.set_fontsize(15)
    axs[0].set_ylabel("Sepal width (cm)")
    axs[0].xaxis.label.set_fontsize(15)
    axs[0].set_xlabel("Sepal length (cm)")

    axs[1].scatter(
        np.array(versicolor[iris_dataframe.columns[2]]),
        np.array(versicolor[iris_dataframe.columns[3]]),
        c='b',
        label='Versicolor'
    )
    axs[1].scatter(
        np.array(setosa[iris_dataframe.columns[2]]),
        np.array(setosa[iris_dataframe.columns[3]]),
        c='r',
        label='Setosa'
    )
    axs[1].scatter(
        np.array(virginica[iris_dataframe.columns[2]]),
        np.array(virginica[iris_dataframe.columns[3]]),
        c='g',
        label='Virginica'
    )

    axs[1].yaxis.label.set_fontsize(15)
    axs[1].set_ylabel("Petal width (cm)")
    axs[1].xaxis.label.set_fontsize(15)
    axs[1].set_xlabel("Petal length (cm)")

    axs[0].text(3.45, 4.35, 'a)', fontsize=20)
    axs[1].text(-0.3, 2.45, 'b)', fontsize=20)

    plt.legend(loc='lower right')
    plt.savefig("scatter_iris_01.png", dpi=300)


def special_scatter():
    fig, axs = plt.subplots(figsize=(8, 8), facecolor='w', edgecolor='k')
    plt.subplots_adjust(
        left=0.08, right=0.95, top=0.98, bottom=0.08,
        wspace=0.3, hspace=0.3
    )

    axs.scatter(
        np.array(versicolor[iris_dataframe.columns[2]]),
        np.array(versicolor[iris_dataframe.columns[3]]),
        c='b',
        label='Versicolor'
    )
    axs.scatter(
        np.array(setosa[iris_dataframe.columns[2]]),
        np.array(setosa[iris_dataframe.columns[3]]),
        c='r',
        label='Setosa'
    )
    axs.scatter(
        np.array(virginica[iris_dataframe.columns[2]]),
        np.array(virginica[iris_dataframe.columns[3]]),
        c='g',
        label='Virginica'
    )

    # create new axes on the right and top
    divider = make_axes_locatable(axs)
    axs_boxx = divider.append_axes("top", size=3.0, pad=0.1, sharex=axs)
    axs_boxy = divider.append_axes("right", size=3.0, pad=0.1, sharey=axs)

    # hide ticks
    axs_boxx.xaxis.set_tick_params(labelbottom=False)
    axs_boxx.yaxis.set_tick_params(labelleft=False)
    plt.setp(axs_boxx.get_xticklabels(), visible=False)
    axs_boxx.tick_params(axis='both', which='both', length=0)

    axs_boxy.xaxis.set_tick_params(labelbottom=False)
    axs_boxy.yaxis.set_tick_params(labelleft=False)
    plt.setp(axs_boxy.get_xticklabels(), visible=False)
    axs_boxy.tick_params(axis='both', which='both', length=0)

    # boxplot X
    data = [
        setosa[iris_dataframe.columns[2]],
        versicolor[iris_dataframe.columns[2]],
        virginica[iris_dataframe.columns[2]]
    ]

    bp = axs_boxx.boxplot(
        data,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(alpha=0.6),
        orientation="horizontal"
    )

    colors = ['r', 'b', 'g']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.setp(bp['medians'], color='k')

    # boxplot Y
    data = [
        setosa[iris_dataframe.columns[3]],
        versicolor[iris_dataframe.columns[3]],
        virginica[iris_dataframe.columns[3]]
    ]

    bp = axs_boxy.boxplot(
        data,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(alpha=0.6)
    )

    colors = ['r', 'b', 'g']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    plt.setp(bp['medians'], color='k')

    axs.yaxis.label.set_fontsize(15)
    axs.set_ylabel("Petal width (cm)")
    axs.xaxis.label.set_fontsize(15)
    axs.set_xlabel("Petal length (cm)")

    first_leg = mpatches.Patch(color='red', label='Setosa')
    second_leg = mpatches.Patch(color='blue', label='Versicolor')
    third_leg = mpatches.Patch(color='green', label='Virginica')

    plt.legend(
        handles=[first_leg, second_leg, third_leg],
        bbox_to_anchor=(1.0, 1.75)
    )

    plt.savefig("special_scatter_iris_01.png", dpi=300)


if __name__ == '__main__':
    scatter_1d()
    scatter_2d()
    special_scatter()
