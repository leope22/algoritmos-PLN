'''
EX01 - Linguística e Pré-processamento

Nesta atividade você irá implementar algumas técnicas básicas de de pré-processamento de texto usando duas bibliotecas principais: NLTK e spaCy.

Você deverá completar nove funções:

normalizaEmails(): responsável por normalizar a ocorrência de endereços de e-mail no texto;
removeTagsHTML(): responsável por remover qualquer tag HTML do texto;
normalizaNumeros(): responsável por normalizar a ocorrência de números no texto;
removeEspacosExtras(): responsável por remover espaços em branco adicionais presentes em um texto;
removeStopwordsNLTK(): responsável por remover stopwords da língua portuguesa presentes no texto;
realizaStemmingNLTK(): responsável por realizar a "stemização" de palavras e reduzí-las para a sua base ou raiz;
preprocessamentoDeTextoNLTK(): responsável por aplicar diversas etapas de pré-processamento de texto com recursos da NLTK;
imprimePOSTaggingSpacy(): responsável por realizar o processamento do texto com spaCy e imprimir informações de POS Tagging;
imprimeEntidadesSpacy(): responsável por realizar o processamento do texto com spaCy e imprimir informações de entidades.
'''

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

def normalizaEmails( texto: str ) -> str:
    """
    Normaliza a ocorrência de endereços de e-mail no texto. Nesse momento, 
    considere que os e-mails possuem o formato <usuario>@<dominio>.<extensao>,
    sem a presença de espaços em quaisquer um dos componentes anteriores.
    """
    email_normalizado = "EMAIL"

    padrao_email = r'\b[A-Za-z0-9._-]+@[A-Za-z0-9.-]+(?:\.[A-Za-z])?[.,;]?'
    texto = re.sub(padrao_email, email_normalizado, texto)

    return texto