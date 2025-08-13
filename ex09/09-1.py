'''
EX09 - Tradução Automática

Você deverá completar 5 questões, algumas delas com funções que pertencentes a determinadas classes:

traduzir_palavra_a_palavra: responsável por calcular a matriz de covariância e encontrar os autovalores e autovetores;
avaliar_traducao: responsável por utilizar os autovetores encontrados para projetar os dados em um espaço de dimensão reduzida;
classeVocabulary: responsável por implementar um vocabulário de um Neural Machine Translation (NMT) model;
classeNMTEncoder: responsável por implementar um codificador de um Neural Machine Translation (NMT) model;
classeNMTDecoder: responsável por implementar um decodificador de um Neural Machine Translation (NMT) model.
'''

def traduzir_palavra_a_palavra(entrada: dict):
    """
    Traduz frases de português para inglês usando um dicionário fornecido na entrada.
    Compara a tradução gerada com a referência e calcula acurácia.

    Parâmetros:
    entrada: dict com:
        - "frase_pt": frase em português (str)
        - "referencia_en": frase esperada em inglês (list[str])
        - "dicionario": dicionário de tradução palavra-a-palavra (dict)

    Retorno:
    dict com:
        - "traducao_gerada": lista de palavras traduzidas
        - "referencia": lista de palavras alvo
        - "acuracia": percentual de palavras corretas
    """

    frase_pt = entrada.get("frase_pt", "").lower().split()
    referencia_en = entrada.get("referencia_en", [])
    dicionario = entrada.get("dicionario", {})
    traducao_gerada = []

    for palavra in frase_pt:
        traducao = dicionario.get(palavra, "[UNK]")
        traducao_gerada.append(traducao)
    
    corretas = 0
    min_len = min(len(traducao_gerada), len(referencia_en))
    for i in range(min_len):
        if traducao_gerada[i] == referencia_en[i]:
            corretas += 1
    acuracia = corretas / len(referencia_en) if len(referencia_en) > 0 else 0.0

    return {
        "frase_pt": frase_pt,
        "traducao_gerada": traducao_gerada,
        "referencia": referencia_en,
        "acuracia": acuracia
    }