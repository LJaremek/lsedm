from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
from pyvis.network import Network
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np


def get_correlation_matrix(
        df: pd.DataFrame,
        method: str = "spearman"
        ) -> pd.DataFrame:

    method = method.lower()

    if method == "pearson":
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
        return scaled_df.corr()
    elif method == "spearman":
        return df.corr(method="spearman")
    else:
        msg = f"Unknown correlation method '{method}'"
        msg += "Available method: 'pearson', 'spearman'"
        raise Exception(msg)


def plot_correlation_matrix(
        df_corr: pd.DataFrame,
        fig_size: tuple[int, int] | None = None,
        exclude_upper_triangle: bool = True,
        exclude_diagonal: bool = True,
        exclude_min: bool = True
        ) -> None:

    title = "Correlation Matrix"

    if exclude_upper_triangle:
        title += "\nwith excluded upper triangle"
        mask = pd.DataFrame(
            True,
            index=df_corr.index,
            columns=df_corr.columns
            )
        mask.values[np.triu_indices_from(mask)] = False
        corr_matrix = df_corr.where(mask)

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


def cluster_correlation_matrix(
        corr: pd.DataFrame,
        method: str = "average"
        ) -> pd.DataFrame:
    """
    Klastrowanie macierzy korelacji przy użyciu hierarchicznego klastrowania.
    Odległość definiujemy jako 1 - |korelacja|,
        a następnie przestawiamy wiersze i kolumny.
    """
    distance_matrix = 1 - corr.abs()

    condensed_distance = squareform(distance_matrix)

    linkage_matrix = sch.linkage(condensed_distance, method=method)

    idx = sch.leaves_list(linkage_matrix)

    clustered_corr = corr.iloc[idx, :].iloc[:, idx]
    return clustered_corr


def plot_dendrogram(
        corr: pd.DataFrame,
        method: str = "average",
        orientation: str = "left",
        fig_size: tuple[int, int] = None
        ) -> None:
    """
    Rysuje dendrogram na podstawie macierzy korelacji
    z możliwością zmiany orientacji.

    Parametry:
    - corr: Macierz korelacji (DataFrame)
    - method: Metoda łączenia w hierarchicznym klastrowaniu
    - orientation: Orientacja dendrogramu ('top', 'bottom', 'left', 'right')
    - fig_size: Rozmiar wykresu
    """
    if fig_size is None:
        fig_size = (12, 8)

    distance_matrix = 1 - corr.abs()
    condensed_distance = squareform(distance_matrix)

    linkage_matrix = sch.linkage(condensed_distance, method=method)

    plt.figure(figsize=fig_size)
    sch.dendrogram(
        linkage_matrix,
        labels=corr.columns,
        orientation=orientation
        )
    plt.title("Dendrogram")
    plt.show()


def build_network(
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.5
        ) -> tuple[nx.Graph, list[frozenset]]:

    G = nx.Graph()
    triu_indices = np.triu_indices_from(correlation_matrix, k=1)
    for i, j in zip(*triu_indices):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > threshold:
            G.add_edge(
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                weight=corr_value,
                width=abs(corr_value) * 5,
                title=f"Correlation: {corr_value:.2f}"
            )

    communities = list(
        nx.algorithms.community.greedy_modularity_communities(G)
        )

    for group_id, community in enumerate(communities):
        for node in community:
            G.nodes[node]["group"] = group_id

    centrality = nx.degree_centrality(G)
    for node, cent in centrality.items():
        G.nodes[node]["value"] = cent * 50
    return G, communities


def visualize_network(
        correlation_matrix: pd.DataFrame,
        file_name: str = "correlation_network.html",
        threshold: float = 0.5
        ) -> list[frozenset]:

    G, communities = build_network(correlation_matrix, threshold)
    net = Network(
        height="750px",
        width="100%",
        cdn_resources="in_line",
        bgcolor="#222222",
        font_color="white",
        notebook=True
    )
    net.from_nx(G)
    net.show_buttons(filter_=["nodes", "edges", "physics"])
    net.save_graph(file_name)
    return communities
