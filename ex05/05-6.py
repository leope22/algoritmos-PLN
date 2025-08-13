import nltk

try:
    nltk.data.find('corpora/treebank')

except LookupError:
    nltk.download('treebank', quiet=True)

from nltk.corpus import treebank
from nltk.tree import Tree
from typing import List

def extrairSentencasPorEstrutura( estrutura: str, num_sentencas: int = 3 ) -> List[str]:
    """
    Retorna as primeiras sentenças do corpus Treebank que satisfazem uma estrutura sintática específica.
    Estruturas disponíveis: 'NP_VP_NP', 'S_PP'
    """
    
    sentencas_encontradas = []
    arvores = treebank.parsed_sents()[0:2300]

    for arvore in arvores:
        if estrutura == 'NP_VP_NP':
            if len(arvore) >= 2 and arvore.label() == 'S':
                if arvore[0].label() == 'NP' and arvore[1].label() == 'VP':
                    for filho in arvore[1]:
                        if isinstance(filho, Tree) and filho.label() == 'NP':
                            sentencas_encontradas.append(" ".join(arvore.leaves()))
                            break

        elif estrutura == 'S_PP':
            if any(subarvore.label() == 'PP' for subarvore in arvore.subtrees()):
                sentencas_encontradas.append(" ".join(arvore.leaves()))

        if len(sentencas_encontradas) >= num_sentencas:
            break

    return sentencas_encontradas