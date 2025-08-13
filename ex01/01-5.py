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

def removeStopwordsNLTK( texto: str ) -> str:
    """
    Remove stopwords da língua portuguesa presentes no texto.
    """
    palavras = word_tokenize(texto, language='portuguese')
    stop_words = set(stopwords.words('portuguese'))
    palavra_excecao = {"Embora"}
    
    palavras_filtradas = [
        palavra for palavra in palavras
        if (palavra.lower() not in stop_words or palavra in palavra_excecao) 
        and (palavra.isalpha() or palavra in {',', '.', '!'})
    ]
    
    texto = ' '.join(palavras_filtradas)

    return texto