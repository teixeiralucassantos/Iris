import matplotlib.pyplot as plt
import numpy as np

# Importação necessária para a projeção 3D no matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
from sklearn import datasets
from sklearn.cluster import KMeans

# Configurar a semente aleatória para reprodutibilidade
np.random.seed(5)

# Carregar o conjunto de dados Iris
dados_iris = datasets.load_iris()
X = dados_iris.data
y = dados_iris.target

# Definir diferentes estimadores de K-Means
kmeans_variants = [
    ("k_means_iris_8_clusters", KMeans(n_clusters=8)),
    ("k_means_iris_3_clusters", KMeans(n_clusters=3)),
    ("k_means_iris_bad_initialization", KMeans(n_clusters=3, n_init=1, init="random")),
]

# Criar uma figura para os gráficos
figura = plt.figure(figsize=(10, 8))
titulos = ["8 Clusters", "3 Clusters", "3 Clusters, Inicialização Ruim"]

# Loop através dos estimadores e seus títulos
for indice, ((nome, estimador), titulo) in enumerate(zip(kmeans_variants, titulos)):
    eixos = figura.add_subplot(2, 2, indice + 1, projection="3d", elev=48, azim=134)
    estimador.fit(X)
    rotulos = estimador.labels_

    # Plotar os pontos de dados
    eixos.scatter(X[:, 3], X[:, 0], X[:, 2], c=rotulos.astype(float), edgecolor="k")

    # Remover rótulos dos eixos
    eixos.xaxis.set_ticklabels([])
    eixos.yaxis.set_ticklabels([])
    eixos.zaxis.set_ticklabels([])
    eixos.set_xlabel("Largura da Pétala")
    eixos.set_ylabel("Comprimento do Sépalo")
    eixos.set_zlabel("Comprimento da Pétala")
    eixos.set_title(titulo)

# Plotar a verdade de base
eixos_verdade = figura.add_subplot(2, 2, 4, projection="3d", elev=48, azim=134)

# Adicionar texto para cada classe
for nome_classe, rotulo in [("Setosa", 0), ("Versicolor", 1), ("Virginica", 2)]:
    eixos_verdade.text3D(
        X[y == rotulo, 3].mean(),
        X[y == rotulo, 0].mean(),
        X[y == rotulo, 2].mean() + 2,
        nome_classe,
        horizontalalignment="center",
        bbox=dict(alpha=0.2, edgecolor="w", facecolor="w"),
    )

# Plotar os dados reais
eixos_verdade.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor="k")

# Remover rótulos dos eixos
eixos_verdade.xaxis.set_ticklabels([])
eixos_verdade.yaxis.set_ticklabels([])
eixos_verdade.zaxis.set_ticklabels([])
eixos_verdade.set_xlabel("Largura da Pétala")
eixos_verdade.set_ylabel("Comprimento do Sépalo")
eixos_verdade.set_zlabel("Comprimento da Pétala")
eixos_verdade.set_title("Verdade de Base")

# Ajustar espaço entre subgráficos
plt.subplots_adjust(wspace=0.25, hspace=0.25)
plt.show()