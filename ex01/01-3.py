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

def normalizaNumeros( texto: str ) -> str:
    """
    Normaliza a ocorrência de números no texto. Nesse momento, considere números
    inteiros ou decimais que incluam a parte inteira.
    """
    numero_normalizado = "NUMERO"
    
    padrao = r'\b(\d+)(\.\d*)?\b'
    texto = re.sub(padrao, numero_normalizado, texto)

    return texto