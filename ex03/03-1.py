'''
EX03 - Representação de Texto: Parte 2

Nesta atividade você irá implementar algumas operações básicas de representação de texto.

Você deverá completar cinco funções:

similaridadeWord2Vec(): responsável por criar um objeto do tipo Word2Vec a partir de um corpus (lista de strings) bruto, retornando seu vocabulario e top-3 termos mais similares a palavra-alvo;
criaVetorDocumento(): responsável por criar um vetor para um novo documento a partir de um modelo ja existente calculando-se a media dos vetores de cada palavra. Caso alguma palavra individual do documento nao possua um vetor proprio, ignore-a.;
documentosSimilares(): responsável por calcular, a partir de um texto de entrada, sua similaridade a um conjunto pré-definido de documentos. Retorna uma lista com os índices dos top-n documentos mais semelhantes ao texto de entrada. Utilize a função de pré-processamento preprocessamentoDeTextoNLTK já implementada, assim como a função criaVetorDocumento. Utilize também recursos da scikit-learn.;
criaMatrizSimilaridade(): responsável por gerar embeddings, utilizando modelo de linguagem pré-existente, para as frases de entrada e calcular a matriz de similaridade de cosseno entre elas.  Utilize recursos da scikit-learn; e
comparaEmbeddingsContextuais(): responsável por gerar embeddings de frases distintas que contenham uma palavra em comum e, com isso, calcular a similaridade de cossenos dessas embeddings para entender a geração de embeddings com uma mesma palavra em contextos distintos. 

Algumas das funções a serem implementadas nesta atividade utilizam recursos auxiliares previamente implementados, veja o protótipo desses recursos abaixo:

WORD2VEC_CORPUS = List[str]
function preprocessamentoDeTextoNLTK( texto: str ) -> str
function criaVetorDocumento( documento: str ): -> numpy.ndarray
function criaModeloWord2Vec(): -> gensim.models.Word2Vec
'''

from auxiliar import WORD2VEC_CORPUS, preprocessamentoDeTextoNLTK
import gensim
from gensim.models import Word2Vec

import numpy as np

def similaridadeWord2Vec( palavra_alvo: str ):
    """
    Cria um objeto do tipo Word2Vec a partir de um corpus (lista de strings) bruto, retornando seu vocabulario e top-3 termos mais similares a palavra-alvo.
    """

    TAMANHO_DO_VETOR = 200
    MIN_OCORRENCIAS = 1
    vocabulario = None
    similares = []

    corpus_preprocessado = []
    for texto in WORD2VEC_CORPUS:
        texto_processado = preprocessamentoDeTextoNLTK(texto)
        corpus_preprocessado.append(texto_processado.split())

    model = Word2Vec(
        sentences=corpus_preprocessado,
        vector_size=TAMANHO_DO_VETOR,
        min_count=MIN_OCORRENCIAS,
        workers=4
    )

    vocabulario = list(model.wv.key_to_index.keys())

    if palavra_alvo in vocabulario:
        similares = model.wv.most_similar(palavra_alvo, topn=3)

    return vocabulario, similares