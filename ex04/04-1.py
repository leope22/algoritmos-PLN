'''
EX04 - PoS Tagging & REN

Nesta atividade você irá implementar algumas operações básicas de etiquetação morfossintática e reconhecimento de entidades nomeadas.

Você deverá completar quatro funções:

contaPosTags(): responsável por contar a frequência de cada PoS tag nos documentos em português usando spaCy;
criaHMM(): responsável por retornar um modelo HMM treinado de forma supervisionada sobre os dados de treinamento, usando recursos da NLTK;
calculaProbabilidadeSequencia(): responsável por realizar o tageamento de um texto e, além do própria tageamento, também retorna a probabilidade de ocorência daquela sequência; e
extraiFeatures(): responsável por extrair características avançadas de cada token de uma sentença em português com spaCy. As características de cada token devem ser agrupadas em dicionários que devem ser retornados juntos. As características incluem PoS, lemas, dependências, contexto expandido, NER e relações sintáticas.

Algumas das funções a serem implementadas nesta atividade utilizam recursos auxiliares previamente implementados, veja o protótipo desses recursos abaixo:

WORD2VEC_CORPUS = List[str]
'''

from auxiliar import *

import spacy
from collections import Counter
from typing import List, Dict

def contaPosTags(documentos: List[str]) -> Dict[str, int]:
    """
    Conta a frequência de cada PoS tag nos documentos em português usando spaCy.
    """

    pipeline = spacy.load('pt_core_news_sm')

    contador = Counter()
    
    for doc in documentos:
        processed = pipeline(doc)
        for token in processed:
            contador[token.pos_] += 1
    
    return sorted(contador.items(), key=lambda item: item[1], reverse=True)

def exibirTabelaTags(freq_tags: Dict[str, int]) -> None:
    pos_descricao = {
        'ADJ': 'Adjetivo',
        'ADP': 'Preposição',
        'ADV': 'Advérbio',
        'AUX': 'Verbo auxiliar',
        'CCONJ': 'Conjunção coordenativa',
        'DET': 'Determ. / Artigo',
        'INTJ': 'Interjeição',
        'NOUN': 'Substantivo',
        'NUM': 'Número',
        'PART': 'Partícula',
        'PRON': 'Pronome',
        'PROPN': 'Nome próprio',
        'PUNCT': 'Pontuação',
        'SCONJ': 'Conjunção subordinativa',
        'SYM': 'Símbolo',
        'VERB': 'Verbo principal',
        'X': 'Outro',
        'SPACE': 'Espaço'
    }

    print(f"{'+':-<44}+")
    print(f"| {'POS Tag':<10} | {'Categoria':<24} | {'Freq.':>5} |")
    print(f"{'+':-<44}+")

    for tag, freq in freq_tags:
        descricao = pos_descricao.get(tag, 'Desconhecida')
        print(f"| {tag:<10} | {descricao:<24} | {freq:>5} |")

    print(f"{'+':-<44}+")