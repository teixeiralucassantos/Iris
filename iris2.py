import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn import datasets
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

# Carregar o conjunto de dados Iris
dados_iris = datasets.load_iris()
# Usar as últimas duas características do conjunto de dados para visualização
X = dados_iris.data[:, 2:4]  
# As classes (targets) do conjunto de dados
y = dados_iris.target

# Definindo o número de características
num_features = X.shape[1]

# Configuração do parâmetro de regularização para os classificadores
regularizacao = 10
# Definindo o kernel para o Classificador de Processo Gaussiano
kernel_func = 1.0 * RBF([1.0, 1.0])  

# Inicializando diferentes classificadores
classificadores = {
    "Regressão Logística L1": LogisticRegression(C=regularizacao, penalty="l1", solver="saga", max_iter=10000),
    "Regressão Logística L2 (Multinomial)": LogisticRegression(
        C=regularizacao, penalty="l2", solver="saga", max_iter=10000
    ),
    "Regressão Logística L2 (OvR)": OneVsRestClassifier(
        LogisticRegression(C=regularizacao, penalty="l2", solver="saga", max_iter=10000)
    ),
    "SVC Linear": SVC(kernel="linear", C=regularizacao, probability=True, random_state=0),
    "Classificador GPC": GaussianProcessClassifier(kernel_func),
}

# Número de classificadores
num_classificadores = len(classificadores)

# Criando subplots para cada classificador
figura, eixos = plt.subplots(
    nrows=num_classificadores,
    ncols=len(dados_iris.target_names),
    figsize=(3 * 2, num_classificadores * 2.5),  # Aumentei a altura
)

# Loop através de cada classificador
for idx_classifier, (nome, classificador) in enumerate(classificadores.items()):
    # Ajusta o classificador aos dados e faz previsões
    y_previsto = classificador.fit(X, y).predict(X)
    # Calcula a precisão do modelo nos dados de treinamento
    precisao = accuracy_score(y, y_previsto)
    print(f"Precisão (treino) para {nome}: {precisao:0.1%}")
    
    # Loop através de cada classe
    for classe in np.unique(y):
        # Exibir a estimativa de probabilidade do classificador
        disp = DecisionBoundaryDisplay.from_estimator(
            classificador,
            X,
            response_method="predict_proba",
            class_of_interest=classe,
            ax=eixos[idx_classifier, classe],
            vmin=0,
            vmax=1,
        )
        # Definindo o título do gráfico com o nome da classe
        eixos[idx_classifier, classe].set_title(f"Classe {['Setosa', 'Versicolor', 'Virginica'][classe]}", fontsize=10)
        
        # Filtrar dados previstos para a classe atual
        mascara_y_previsto = y_previsto == classe
        eixos[idx_classifier, classe].scatter(
            X[mascara_y_previsto, 0], X[mascara_y_previsto, 1], marker="o", c="w", edgecolor="k"
        )
        eixos[idx_classifier, classe].set(xticks=(), yticks=())
        
    # Ajuste do rótulo do eixo y para melhor visibilidade com fonte menor
    eixos[idx_classifier, 0].set_ylabel(nome, fontsize=8, labelpad=20)  # Diminuí o tamanho da fonte

# Adicionando uma barra de cores para representar as probabilidades
ax_color = plt.axes([0.15, 0.04, 0.7, 0.02])
plt.title("Probabilidade")
_ = plt.colorbar(
    cm.ScalarMappable(norm=None, cmap="viridis"), cax=ax_color, orientation="horizontal"
)

# Melhorar o layout do gráfico para evitar sobreposição
plt.tight_layout(pad=3.0)  # Aumentei o padding para evitar sobreposição
plt.show()
