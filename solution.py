import pandas as pd
import numpy as np

def calcular_regressao(arquivo):
    # 1. Ler o arquivo CSV
    df = pd.read_csv(arquivo)
    # 2. Separar as variáveis de entrada e o resultado
    X = df[["area_util_m2", "quantidade_quartos"]].to_numpy()
    y = df["preco_venda"].to_numpy()
    # 3. Adicionar uma coluna de 1 para calcular o beta0
    X = np.column_stack((np.ones(len(X)), X))
    # 4. Calcular os coeficientes da regressão
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    # 5. Retornar os coeficientes como uma lista
    return beta.tolist()

# Executar a função
resultado = calcular_regressao("imoveis_lar_perfeito.csv")
print(resultado)