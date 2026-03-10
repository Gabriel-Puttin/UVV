import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

iris_dataset = load_iris()
iris_dataframe = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)

iris_dataframe['Iris type'] = iris_dataset['target']
iris_dataframe['Species'] = iris_dataframe['Iris type'].apply(lambda x: 'setosa' if
x == 0 else ('versicolor' if x == 1 else 'virginica'))

versicolor = iris_dataframe.loc[iris_dataframe['Species'] == "versicolor"]
setosa = iris_dataframe.loc[iris_dataframe['Species'] == "setosa"]
virginica = iris_dataframe.loc[iris_dataframe['Species'] == "virginica"]


def create_boxplot():
    fig, axs = plt.subplots(2, 2, figsize=(15, 20), facecolor='w', edgecolor='k')
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
                "Boxplot for {}".format(iris_dataframe.columns[a]),
                fontsize=25
            )

            # Convert to a list of series
            data = [
                versicolor[iris_dataframe.columns[a]],
                setosa[iris_dataframe.columns[a]],
                virginica[iris_dataframe.columns[a]]
            ]

            bp = axa.boxplot(
                data,
                widths=0.6,
                patch_artist=True,
                boxprops=dict(alpha=0.6)
            )

            xticklabels = ['Versicolor', 'Setosa', 'Virginica']
            axa.set_xticklabels(xticklabels)

            colors = ['b', 'r', 'g']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)

            plt.setp(bp['medians'], color='k')

            axa.tick_params(axis='y', labelsize=15)
            axa.yaxis.label.set_fontsize(20)
            axa.set_ylabel("Values")

            axa.tick_params(axis='x', labelsize=20)

            a += 1

    axs[0, 0].text(0, 7.90, 'a)', fontsize=25)
    axs[0, 1].text(0, 4.42, 'b)', fontsize=25)
    axs[1, 0].text(0, 6.90, 'c)', fontsize=25)
    axs[1, 1].text(0, 2.50, 'd)', fontsize=25)

    plt.savefig("boxplot_iris_01.png", dpi=300)


if __name__ == '__main__':
    create_boxplot()
