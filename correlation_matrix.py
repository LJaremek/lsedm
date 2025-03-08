from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
    return scaled_df


def draw_correlation_matrix(
        df: pd.DataFrame,
        fig_size: tuple[int, int] | None = None,
        exclude_upper_triangle: bool = False,
        exclude_diagonal: bool = False,
        exclude_min: bool = False
        ) -> None:
    corr_matrix = df.corr()

    title = "Correlation Matrix"

    if exclude_upper_triangle:
        title += "\nwith excluded upper triangle"
        mask = pd.DataFrame(
            True,
            index=corr_matrix.index,
            columns=corr_matrix.columns
            )
        mask.values[np.triu_indices_from(mask)] = False
        corr_matrix = corr_matrix.where(mask)

    if exclude_diagonal:
        title += "\nwith excluded diagonal"
        for i in range(len(corr_matrix)):
            corr_matrix.iloc[i, i] = np.nan

    if exclude_min:
        title += "\nwith excluded minimum values"
        min_value = corr_matrix.min().min()
        corr_matrix.replace(min_value, np.nan, inplace=True)

    if fig_size is None:
        fig_size = (12, 8)
    plt.figure(figsize=fig_size)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")

    plt.title(title)
    plt.show()
