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

def realizaStemmingNLTK( texto: str, termos_normalizados: list = ['EMAIL', 'NUMERO']) -> str:
    """
    Aplica o processo de stemming em um texto. Os termos presentes em termos_normalizados não devem sofrer stemming.
    """
    stemmer = RSLPStemmer()
    
    stop_words = set(stopwords.words('portuguese'))
    tokens = re.findall(r'\w+(?:-\w+)*|\s+|[^\w\s]', texto)

    resultado = ''
    for i, token in enumerate(tokens):
        if re.fullmatch(r'\s+', token):
            resultado += token
        elif re.fullmatch(r'[^\w\s]', token):
            if not resultado.endswith(' '):
                resultado += ' '
            resultado += token
        elif token.upper() in termos_normalizados:
            resultado += token
        else:
            resultado += stemmer.stem(token.lower())
    texto = resultado

    return texto