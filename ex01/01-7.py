import re, nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('toeknizers/punkt_tab')
    nltk.data.find('stemmers/rslp')

except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('rslp', quiet=True)

def preprocessamentoDeTextoNLTK( texto: str ) -> str:
    """
    Responsavel por aplicar diversas etapas de pre-processamento de texto com recursos da NLTK. Nesta função, invoque as funções auxiliares implementadas nas questões anteriores.
    O texto deve ter estruturas de HTML removidas, normalização de certos padrões, retirada de palavras redudantes e radicalização (stemming).
    """
    texto = removeTagsHTML(texto)
    texto = normalizaEmails(texto)
    texto = normalizaNumeros(texto)
    texto = removeStopwordsNLTK(texto)
    texto = realizaStemmingNLTK(texto)
    texto = removeEspacosExtras(texto)

    return texto