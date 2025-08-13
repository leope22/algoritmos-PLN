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

def removeTagsHTML( texto: str ) -> str:
    """
    Remove qualquer tag HTML do texto, substituindo por um espaço em branco.
    """
    texto = re.sub(r'<!--.*?-->', ' ', texto)
    texto = re.sub(r'<[^>]+>', ' ', texto)
    texto = texto.strip()
    
    return texto