'''
EX02 - Representação de Texto: Parte 1

Nesta atividade você irá implementar algumas operações básicas de representação de texto.

Você deverá completar cinco funções:

criaRepresentacaoOneHot(): responsável por criar representações one hot para o texto de entrada, o qual já está tokenizado. Utilize recursos da scikit-learn;
criaMatrizTermoDocumento(): responsável por criar a matriz de frequência termo-documento utilizando recursos da scikit-learn;
criaRepresentacaoTFIDF(): responsável por criar representações TFIDF para o texto de entrada, o qual já está tokenizado. Utilize recursos da scikit-learn;
criaWordCloud(): responsável por gerar uma nuvem de palavras com base no texto já tokenizado e retornar a frequência das 3 palavras mais frequentes;
criaMatrizPMI(): responsável por gerar uma matriz de PMI com base em coocorrências no texto já tokenizado e retornar a matriz gerada. Utilize recursos da scikit-learn.
'''

from sklearn.preprocessing import OneHotEncoder

def criaRepresentacaoOneHot( texto_tokenizado: list[list[str]] ):
    """
    Gera uma representacao one hot para o texto tokenizado recebido como entrada usando recursos da scikit-learn.
    """

    enc = OneHotEncoder(handle_unknown='ignore')
    vocab, vetores = None, None

    vetores = enc.fit_transform(texto_tokenizado)
    vocab = enc.get_feature_names_out()
    vocab = [word.replace("x0_", "") for word in enc.get_feature_names_out()]

    return vocab, vetores.toarray()