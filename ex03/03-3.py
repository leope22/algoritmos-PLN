import sklearn
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

from auxiliar import WORD2VEC_CORPUS, preprocessamentoDeTextoNLTK, criaVetorDocumento

def documentosSimilares( texto: str, topn: int = 2 ):
    """
    A partir de um texto de entrada, calcule sua similaridade a um conjunto pré-definido de documentos. Retorne uma lista com os índices dos top-n documentos mais semelhantes ao texto de entrada. 
    Utilize a função de pré-processamento preprocessamentoDeTextoNLTK já implementada, assim como a função criaVetorDocumento. Utilize também recursos da scikit-learn.
    """

    X = []
    for doc in WORD2VEC_CORPUS:
        X.append(criaVetorDocumento(doc))
    X = np.array(X)

    indices_documentos_semelhantes = []

    vetor_texto = criaVetorDocumento(texto).reshape(1, -1)

    similaridades = cosine_similarity(vetor_texto, X)

    indices_ordenados = np.argsort(similaridades[0])[::-1]

    indices_documentos_semelhantes = indices_ordenados.tolist()

    return indices_documentos_semelhantes[0:topn]