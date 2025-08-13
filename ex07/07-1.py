'''
EX07 - Categorização de Textos

Nesta atividade, você deverá implementar funções para auxiliar na tarefa de categorização de texto. Mais especificamente, categorização aplicada à análise de sentimentos e detecção de notícias falsas. 

Você deverá completar seis funções:

treinaNaiveBayes: responsável por realizar o treinamento manual de um classificador Naive Bayes;
classificadorUsandoLexicos: responsável por classificar um texto de entrada com base em no conjunto de léxicos, em português, da Wordnet Affect BR;
classificar_sentimento_por_embeddings: responsável por implementar um classificador de polaridade (positiva ou negativa) de uma sentença, com base na similaridade de embeddings gerados por um modelo contextual;
classificaTweets: responsável por realizar inferência e cálculo de acurácia a partir de subconjuntos de uma base de dados sobre discurso de ódio;
detectar_fake_news_por_embeddings: responsável por realizar a detecção simplificada de notícias falsas utilizando embeddings gerados por um modelo contextual e um classificador K-Nearest Neighbors (KNN);
simular_propagacao_fake_news: responsável por implementar uma simulação da propagação de uma notícia, que pode ser real ou falsa, em uma rede de usuários representada como um grafo.
'''

import math
from collections import defaultdict, Counter
from auxiliar import preprocessamentoDeTextoNLTK

def treinaNaiveBayes( entrada: dict):
    """
    Treinamento manual de um classificador Naive Bayes multinomial a partir de documentos brutos. Use a funcao de pre-processamento
    já implementada preprocessamentoDeTextoNLTK.

    Parâmetros:
    -----------
    entrada : dict
        Dicionario que contem as chaves:
            "docs": lista de documentos, onde cada documento é uma tupla (doc, classe), sendo doc uma string e classe uma string indicando a classe do documento.
            "classes": lista das classes possíveis.

    Retorna:
    --------
    V : set
        Vocabulário — conjunto de todas as palavras presentes no conjunto de treinamento D.
    logprior : dict
        Dicionário onde logprior[c] é o logaritmo da probabilidade a priori da classe c.
    loglikelihood : dict
        Dicionário onde loglikelihood[(w, c)] é o logaritmo da probabilidade de w dado c (log P(w|c)).
    """
    D = entrada['docs']
    C = entrada['classes']
    
    Ndoc = len(D)
    logprior = {}
    loglikelihood = {}
    bigdoc = defaultdict(list)
    V = set()

    docPorClasse = Counter()
    
    for texto, classe in D:
        docPorClasse[classe] += 1
        palavras = preprocessamentoDeTextoNLTK(texto)
        
        if isinstance(palavras, str):
            palavras = palavras.split()
        
        bigdoc[classe].extend(palavras)
        V.update(palavras)
    
    for c in docPorClasse:
        logprior[c] = math.log(docPorClasse[c] / Ndoc)
    
    palavraPorClasse = {c: Counter(bigdoc[c]) for c in docPorClasse}
    totalPalavrasPorClasse = {c: sum(palavraPorClasse[c].values()) for c in docPorClasse}
    V_size = len(V)
    
    for c in docPorClasse:
        for w in V:
            count_wc = palavraPorClasse[c][w]
            loglikelihood[(w, c)] = math.log((count_wc + 1) / (totalPalavrasPorClasse[c] + V_size))
    
    return sorted(V), logprior, loglikelihood