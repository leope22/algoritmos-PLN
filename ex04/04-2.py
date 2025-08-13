import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import nltk

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/treebank')

except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('treebank', quiet=True)

from nltk.tokenize import word_tokenize
from nltk.corpus import treebank
from nltk.tag.hmm import HiddenMarkovModelTrainer

def criaHMM() -> nltk.tag.api.TaggerI:
    """
    Retorna um modelo HMM treinado de forma supervisionada sobre os dados de treinamento.
    """

    sentencas_tageadas_treebank = treebank.tagged_sents()
    dados_treino = sentencas_tageadas_treebank[:3000]

    trainer = HiddenMarkovModelTrainer()
    modelo = trainer.train_supervised(dados_treino)

    return modelo

def calculaProbabilidadeSequencia( modelo, texto: str ):
    """
    Realiza a etiquetação morfossintática de um texto e, além da própria 
    etiquetação, também retorna a probabilidade de ocorência daquela sequência.
    """

    tokens = word_tokenize(texto)
    sentenca_tageada = modelo.tag(tokens)
    
    probabilidade = modelo.probability(sentenca_tageada)

    return sentenca_tageada, probabilidade