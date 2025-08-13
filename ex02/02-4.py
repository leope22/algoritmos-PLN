from sklearn.feature_extraction.text import CountVectorizer

def criaWordCloud(texto_tokenizado: list[list[str]]):
    """
    Mostra as frequências das palavras em um corpus tokenizado.
    """
    palavras_mais_frequentes = None
    quantidade_palavras_frequentes = 3
    
    corpus = [' '.join(tokens) for tokens in texto_tokenizado]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)

    soma_palavras = X.sum(axis=0)

    frequencias = [(palavra, int(soma_palavras[0, idx])) for palavra, idx in vectorizer.vocabulary_.items()]

    frequencias_ordenadas = sorted(frequencias, key=lambda x: -x[1])

    palavras_mais_frequentes = []
    limite = 0
    for palavra, contagem in frequencias_ordenadas:
        if len(palavras_mais_frequentes) < quantidade_palavras_frequentes:
            palavras_mais_frequentes.append((palavra, contagem))
            limite = contagem
        elif contagem == limite:
            palavras_mais_frequentes.append((palavra, contagem))
        else:
            break

    return palavras_mais_frequentes