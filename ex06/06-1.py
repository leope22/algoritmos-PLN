'''
EX06 - Desambiguação Lexical de Sentido (WSD), Anotação de Papéis Semânticos (SRL) & Extração de Informação (IE)

Nesta atividade você irá implementar algumas funções relacionadas às tarefas de Desambiguação Lexical de Sentido (WSD), Anotação de Papéis Semânticos (SRL) & Extração de Informação (IE).

Você deverá completar cinco funções:

desambiguar_wsd: responsável por  implementar uma estratégia prática de Desambiguação de Sentido de Palavras (Word Sense Disambiguation - WSD) utilizando embeddings contextuais;
avaliarWSD: responsável por gerar embeddings para uma lista de sentenças que contenham uma palavra ambígua, calcular a matriz de similaridade de cosseno entre todas as sentenças e retornar essa matriz de similaridade;
detectar_papeis_semanticos: responsável por implementar uma estratégia prática de Rotulagem de Papéis Semânticos (Semantic Role Labeling - SRL) simplificada, baseada na análise das distâncias vetoriais entre o verbo de uma frase e os demais tokens;
comparaVozAtivaVozPassiva: responsável por comparar embeddings de um token específico em duas sentenças — uma na voz ativa e outra na voz passiva — verificando se o modelo é capaz de representar o mesmo papel semântico, independentemente da estrutura sintática; e
avaliaCoberturaNER: responsável por avaliar a cobertura de um modelo de reconhecimento de entidades nomeadas. A cobertura mede se o modelo consegue encontrar as entidades anotadas corretamente nas sentenças. Ela é calculada como a razão entre o número de entidades corretamente encontradas e o total de entidades anotadas no corpus.
'''

import os
import sys

# Redireciona stderr nativamente antes de qualquer import sensível
def bloquear_stderr_nativo():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

bloquear_stderr_nativo()

import numpy as np
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import contextlib

def desambiguar_wsd(entrada: dict):
    """
    Compara embeddings da mesma palavra em dois contextos para verificar similaridade de sentido.
    """

    similaridade = None

    contexto1 = entrada.get("contexto1", "")
    contexto2 = entrada.get("contexto2", "")
    palavra = entrada.get("palavra", "").lower()

    extractor = pipeline(
        "feature-extraction",
        model="prajjwal1/bert-tiny",
        tokenizer="prajjwal1/bert-tiny",
        framework="pt"
    )

    emb1 = extractor(contexto1, return_tensors=False)[0]
    emb2 = extractor(contexto2, return_tensors=False)[0]

    tokenizer = extractor.tokenizer
    tokens1 = tokenizer.tokenize(contexto1)
    tokens2 = tokenizer.tokenize(contexto2)

    def encontrar_posicao(tokens, palavra):
        for i, token in enumerate(tokens):
            if palavra in token.replace("##", ""):
                return i
        return -1

    idx1 = encontrar_posicao(tokens1, palavra)
    idx2 = encontrar_posicao(tokens2, palavra)

    if idx1 == -1 or idx2 == -1:
        return 0.0

    vec1 = np.array(emb1[idx1])
    vec2 = np.array(emb2[idx2])

    similaridade = cosine_similarity([vec1], [vec2])[0][0]

    return similaridade