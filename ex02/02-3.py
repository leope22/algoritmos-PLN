from sklearn.feature_extraction.text import TfidfVectorizer

def criaRepresentacaoTFIDF(texto_tokenizado: list[list[str]]):
    """
    Gera uma representação TF-IDF para o texto tokenizado usando TfidfVectorizer.
    """
    textos = [' '.join(tokens) for tokens in texto_tokenizado]
    
    vectorizer = TfidfVectorizer()
    vetores = vectorizer.fit_transform(textos)
    
    vocab = sorted(vectorizer.vocabulary_.keys())
    
    return vocab, vetores.toarray()