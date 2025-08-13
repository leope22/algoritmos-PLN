from auxiliar import preprocessamentoDeTextoNLTK, criaModeloWord2Vec

import gensim
from gensim.models import Word2Vec

import numpy as np

def criaVetorDocumento( documento: str ):
    """
    Cria um vetor para um novo documento a partir de um modelo ja existente calculando-se a media dos vetores de cada palavra. Caso alguma palavra individual do documento nao possua um vetor proprio, ignore-a.
    """

    modelo = criaModeloWord2Vec()
    vetor_medio = np.zeros(modelo.wv.vector_size)
    documento_processado = preprocessamentoDeTextoNLTK(documento).split()

    palavras_validas = [palavra for palavra in documento_processado 
                       if palavra in modelo.wv.key_to_index]

    if palavras_validas:
        vetores = [modelo.wv[palavra] for palavra in palavras_validas]
        vetor_medio = np.mean(vetores, axis=0)

    return vetor_medio