import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def criaMatrizPMI(texto_tokenizado: list[list[str]]):
    """
    Gera uma matriz de PMI com base em coocorrências.
    """
    matriz = None
    matriz_coocorrencia = None
    contagem_palavras = None
    total_coocorrencia = 0
    
    textos = [' '.join(tokens) for tokens in texto_tokenizado]
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(textos)
    palavras = vectorizer.get_feature_names_out()
    
    matriz_coocorrencia = (X.T @ X).toarray()
    np.fill_diagonal(matriz_coocorrencia, 0)
    
    contagem_palavras = np.array(X.sum(axis=0))[0]
    
    total_coocorrencia = matriz_coocorrencia.sum()
    
    matriz_pmi = np.zeros_like(matriz_coocorrencia, dtype=float)
    for i in range(len(palavras)):
        for j in range(len(palavras)):
            if matriz_coocorrencia[i, j] > 0:
                p_xy = matriz_coocorrencia[i, j] / total_coocorrencia
                p_x = contagem_palavras[i] / X.sum()
                p_y = contagem_palavras[j] / X.sum()
                matriz_pmi[i, j] = max(0, math.log2(p_xy / (p_x * p_y)))
    
    matriz = pd.DataFrame(matriz_pmi, index=palavras, columns=palavras)

    return matriz