# Projeto Iris

Este repositório contém diferentes abordagens para classificar flores utilizando o famoso dataset **Iris** do Scikit-learn. Cada script explora um método de aprendizado de máquina para a classificação em três categorias de flores: **Setosa**, **Versicolor** e **Virginica**.

## Estrutura dos Scripts

1. **iris1.py**: Regressão Simples
    - Neste script, realizo uma **regressão simples** para classificar as flores com base nas características fornecidas no dataset Iris.
    - O modelo prediz a classe da flor de acordo com a relação entre as variáveis independentes e a classe.

2. **iris2.py**: Classificador de Support Vector com Regressão Logística
    - Neste script, utilizo um **classificador de Support Vector** em conjunto com **regressão logística** para realizar a classificação.
    - Este método visa encontrar o hiperplano ótimo que separa as classes de forma eficiente.

3. **iris3.py**: Clusterização com K-Means
    - Aqui, aplico o algoritmo de **clusterização K-Means** para agrupar os dados em três clusters que correspondem às três classes de flores.
    - Apesar de ser um método não supervisionado, conseguimos agrupar as amostras de forma que correspondam bem às classes originais.

## Dataset

O dataset Iris é composto por  amostras de três espécies de flores (**Setosa**, **Versicolor**, **Virginica**), com quatro características por amostra:
- Comprimento da Sépala
- Largura da Sépala
- Comprimento da Pétala
- Largura da Pétala


## Conclusões

Cada script oferece uma maneira diferente de abordar o problema de classificação. A **regressão simples** proporciona uma visão básica do problema, enquanto o **Support Vector com Regressão Logística** explora uma técnica mais avançada. Já o **K-Means** oferece uma abordagem não supervisionada, interessante para comparações com métodos supervisionados.


