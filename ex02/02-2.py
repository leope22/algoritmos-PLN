import nltk
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import CountVectorizer

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('toeknizers/punkt_tab')

except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

def tokenize( texto: str) -> list[str]:
    return nltk.word_tokenize(texto, language="portuguese")

def criaMatrizTermoDocumento( documentos: list[str] ):
    """
    Cria a matriz termo-documento utilizando recursos da scikit-learn.
    Utilize a função de tokenizção definida acima e o parametro
    "token_pattern=True" para evitar warnings. Os termos devem ser
    transformados em mínusculos (lowercase).
    """

    matriz = None
    
    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        lowercase=True,
        token_pattern=None
    )
    
    matriz = vectorizer.fit_transform(documentos)

    return matriz.toarray()