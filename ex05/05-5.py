import spacy
import pt_core_news_sm

def extrairDependenciasSpacy( sentenca: str ):
    """
    """
    
    pipeline = pt_core_news_sm.load()
    dependencias = []
    
    doc = pipeline(sentenca)
    for token in doc:
        if token.dep_ != 'ROOT':
            dependencias.append((token.head.text, token.text, token.dep_))
    
    return dependencias