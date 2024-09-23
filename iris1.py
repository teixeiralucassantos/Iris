import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Carregar o dataset Iris
data = load_iris()
flores = pd.DataFrame(data=data.data, columns=data.feature_names)

# Definindo as variáveis X e y
X = flores[['petal length (cm)']]
y = flores['petal width (cm)']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predições
y_pred = modelo.predict(X_test)

# Criar gráfico de regressão
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Dados reais')
plt.plot(X_test, y_pred, color='red', label='Regressão Linear')
plt.title('Regressão Linear: Petal Length vs Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.grid()
# Salvar gráfico de regressão
plt.savefig(r'C:\Users\User\Documents\portfolio\regressao_linear.png')
plt.close()

# Classificando os dados em grupos
flores['Grupo'] = 'Setosa'  # Grupo 1

# Condição para Versicolor (valores médios)
cond_versicolor = (flores['petal length (cm)'] > flores['petal length (cm)'].quantile(1/3)) & \
                  (flores['petal length (cm)'] <= flores['petal length (cm)'].quantile(2/3)) & \
                  (flores['petal width (cm)'] > flores['petal width (cm)'].quantile(1/3)) & \
                  (flores['petal width (cm)'] <= flores['petal width (cm)'].quantile(2/3))
flores.loc[cond_versicolor, 'Grupo'] = 'Versicolor'  # Grupo 2

# Grupo Virginica (maiores valores)
flores.loc[(flores['petal length (cm)'] > flores['petal length (cm)'].quantile(2/3)) & 
            (flores['petal width (cm)'] > flores['petal width (cm)'].quantile(2/3)), 'Grupo'] = 'Virginica'  # Grupo 3

# Criar gráfico com os grupos
plt.figure(figsize=(10, 6))
for grupo, cor in zip(['Setosa', 'Versicolor', 'Virginica'], ['blue', 'orange', 'green']):
    subset = flores[flores['Grupo'] == grupo]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'], color=cor, label=grupo)

plt.title('Classificação das Flores')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()
plt.grid()
# Salvar gráfico de classificação
plt.savefig(r'C:\Users\User\Documents\portfolio\classificacao_flores.png')
plt.close()
