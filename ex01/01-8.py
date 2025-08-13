import spacy
import pt_core_news_sm

def imprimePOSTaggingSpacy( texto: str ):
    """
    Realiza o processamento do texto de entrada com spaCy, imprime informações de POS Tagging obtidas após o processamento e retorna o texto processado.
    O formato de impressão deve ser o seguinte, para cada token:

        <token>: <token_POSTag>
    """
    doc = None
    pipeline = pt_core_news_sm.load()

    doc = pipeline(texto)
    
    for token in doc:
        print(f"{token.text}: {token.pos_}")