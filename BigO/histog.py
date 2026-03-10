import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.patches as mpatches

iris_dataset = load_iris()
iris_dataframe = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)

iris_dataframe['Iris type'] = iris_dataset['target']
iris_dataframe['Species'] = iris_dataframe['Iris type'].apply(lambda x: 'setosa' if
x == 0 else ('versicolor' if x == 1 else 'virginica'))

versicolor = iris_dataframe.loc[iris_dataframe['Species'] == "versicolor"]
setosa = iris_dataframe.loc[iris_dataframe['Species'] == "setosa"]
virginica = iris_dataframe.loc[iris_dataframe['Species'] == "virginica"]


def histog_kde():
    fig, axs = plt.subplots(2, 2, figsize=(20, 12), facecolor='w', edgecolor='k')
    plt.subplots_adjust(
        left=0.08, right=0.95, top=0.95, bottom=0.06,
        wspace=0.3, hspace=0.3
    )

    a = 0
    name = ['a)', 'b)', 'c)', 'd)']

    for i in range(2):
        for j in range(2):
            axa = axs[i, j]

            axa.set_title(
                "Histogram and PDF of {}".format(iris_dataframe.columns[a]),
                fontsize=20
            )

            axa.hist(
                versicolor[iris_dataframe.columns[a]],
                bins=10, alpha=0.6, color='b'
            )
            axa.axvline(
                versicolor[iris_dataframe.columns[a]].mean(),
                color='b', linestyle='dashed', linewidth=1
            )

            axa.hist(
                setosa[iris_dataframe.columns[a]],
                bins=10, alpha=0.6, color='r'
            )
            axa.axvline(
                setosa[iris_dataframe.columns[a]].mean(),
                color='r', linestyle='dashed', linewidth=1
            )

            axa.hist(
                virginica[iris_dataframe.columns[a]],
                bins=10, alpha=0.6, color='g'
            )
            axa.axvline(
                virginica[iris_dataframe.columns[a]].mean(),
                color='g', linestyle='dashed', linewidth=1
            )

            axs[i, j].locator_params(axis='y', nbins=10)
            axs[i, j].tick_params(axis='y', labelsize=15)
            axs[i, j].yaxis.label.set_fontsize(15)
            axa.set_ylabel("Histogram")

            # Adding Twin Axes to plot KDE
            axa2 = axa.twinx()

            sns.kdeplot(
                versicolor[iris_dataframe.columns[a]],
                fill=True, ax=axa2
            )
            sns.kdeplot(
                setosa[iris_dataframe.columns[a]],
                fill=True, ax=axa2
            )
            sns.kdeplot(
                virginica[iris_dataframe.columns[a]],
                fill=True, ax=axa2
            )

            axa2.tick_params(axis='y', labelsize=15)
            axa2.set_ylim([0, 6])
            axa2.yaxis.label.set_fontsize(15)
            axa2.set_ylabel("KDE")

            axa.set_xlim([-0.25, 9])
            axa.tick_params(axis='x', labelsize=15)
            axa.set_ylim([0, 35])
            axa.xaxis.label.set_fontsize(15)
            axa.set_xlabel("Values")

            axs[i, j].text(-1.5, 33, name[a], fontsize=20)

            a += 1

    first_leg = mpatches.Patch(color='red', label='Setosa')
    second_leg = mpatches.Patch(color='blue', label='Versicolor')
    third_leg = mpatches.Patch(color='green', label='Virginica')

    plt.legend(handles=[first_leg, second_leg, third_leg])
    plt.savefig("histogram_pdf_iris_01.png", dpi=300)


if __name__ == '__main__':
    histog_kde()