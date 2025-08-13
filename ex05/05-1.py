'''
EX05 - Análise Sintática

Nesta atividade você irá implementar algumas funções relacionadas a bases de dados treebank da NLTK, assim como operações de parsing, extração e análise de dependências textuais.

Você deverá completar seis funções:

cky_parse: responsável por verificar se uma sentença é aceita por uma gramática livre de contexto em CNF usando o algoritmo CKY;
simulate_trasition_parser: responsável por simular um parser baseado em transições com SHIFT e REDUCE;
avaliar_uas: responsável por simular uma avaliação de UAS entre heads preditos e padrão-ouro;
validaArvoreDependencia: responsável por determinar se um conjunto de dependências sintáticas forma uma árvore de dependência válida;
extrairDependenciasSpacy: responsável por extrair relações gramaticais de uma sentença e construir a representação de uma árvore de dependência; e
extrairSentencasPorEstrutura: responsável por extrair as três primeiras sentenças da base NLTK treebank que respeitem uma determinada estrutura pré-definida.
'''

def cky_parse(entrada: dict):
    """
    Verifica se uma sentença é aceita por uma gramática livre de contexto em CNF usando o algoritmo CKY.
    """

    aceita = False
    gramatica = entrada.get("gramatica", {})
    sentenca = entrada.get("sentenca", [])

    n = len(sentenca)
    if n == 0:
        return False
    
    tabela = [[set() for _ in range(n+1)] for __ in range(n)]
    
    for i in range(n):
        palavra = sentenca[i]
        for lhs in gramatica:
            for rhs in gramatica[lhs]:
                if isinstance(rhs, str) and rhs == palavra:
                    tabela[i][i+1].add(lhs[0])
    
    for length in range(2, n+1):
        for i in range(n - length + 1):
            j = i + length
            for k in range(i+1, j):
                for lhs in gramatica:
                    for rhs in gramatica[lhs]:
                        if isinstance(rhs, tuple) and len(rhs) == 2:
                            B, C = rhs
                            if B in tabela[i][k] and C in tabela[k][j]:
                                tabela[i][j].add(lhs[0])
    
    aceita = 'S' in tabela[0][n]

    return aceita