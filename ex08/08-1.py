'''
EX08 - Modelos de Linguagem

Você deverá completar três funções:

prever_mascara: responsável por implementar uma função que utiliza um modelo de linguagem do tipo Masked Language Model (MLM) para prever a palavra ausente em uma frase;
classificar_com_prompt_mlm: responsável por implementar uma função que realiza a classificação de sentimento de um texto utilizando um modelo Masked Language Model (MLM);
comparar_impacto_contexto: responsável por implementar uma função cujo objetivo é medir como o contexto influencia uma embedding  de uma palavra em um modelo de linguagem do tipo Transformer.
'''

import os

# Redireciona stderr nativamente
def bloquear_stderr_nativo():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

bloquear_stderr_nativo()

from transformers import pipeline

def prever_mascara(entrada: dict):
    """
    Previsão de palavra mascarada usando modelo MLM pequeno.
    
    Parâmetros:
    entrada: dict com chaves:
        - "frase": string, com token [MASK]
        - "gabarito": string, palavra esperada (sem pontuação)
    
    Retorno:
    dict com:
        - "previsoes": lista das 3 predições
        - "acertou": True/False (se o gabarito está nas predições)
    """
    frase = entrada.get("frase", "")
    gabarito = entrada.get("gabarito", "").lower().strip()
    acertou = False
    palavras_previstas = None

    modelo = pipeline("fill-mask", model="prajjwal1/bert-tiny")

    try:
        resultados = modelo(frase, top_k=3)
        palavras_previstas = [res['token_str'].strip().lower() for res in resultados]
        acertou = gabarito in palavras_previstas
    except:
        palavras_previstas = None
        acertou = False
    
    # Retornar no padrão
    return {
        "previsoes": palavras_previstas,
        "acertou": acertou
    }