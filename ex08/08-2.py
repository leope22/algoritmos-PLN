import os

# Redireciona stderr nativamente
def bloquear_stderr_nativo():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

bloquear_stderr_nativo()

from transformers import pipeline

def classificar_com_prompt_mlm(entrada: dict):
    """
    Classificação como geração via MLM (Masked Language Model).
    
    Parâmetros:
    entrada: dict com chaves:
        - "texto": texto a ser classificado
        - "classe": gabarito esperado ("positivo", "negativo", "indefinido")
    
    Retorno:
    dict com:
        - "previsoes": top 3 palavras previstas para [MASK]
        - "classificacao": categoria (positivo, negativo, indefinido)
        - "gabarito": positivo/negativo
        - "acertou": True/False
    """
    modelo = pipeline("fill-mask", model="prajjwal1/bert-tiny", framework='pt')
    
    texto = entrada.get("texto", "")
    gabarito = entrada.get("classe", "").lower().strip()
    
    # Construir prompt com MASK
    frase_prompt = f"The sentiment of the sentence '{texto}' is [MASK]."

    # Listas de mapeamento para classificação
    positivos = [
        "positive", "good", "great", "excellent", "fantastic", "amazing", "wonderful", "love", "liked", "awesome", "best", "happy", "satisfied"
    ]

    negativos = [
        "negative", "bad", "terrible", "awful", "poor", "worst", "hate", "horrible", "disappointed", "unsatisfied", "angry", "sad"
    ]

    palavras_previstas = []
    classificacao = "indefinido"
    acertou = False

    try:
        resultados = modelo(frase_prompt, top_k=3)
        palavras_previstas = [res['token_str'].strip().lower() for res in resultados]
        
        for palavra in palavras_previstas:
            if palavra in positivos:
                classificacao = "positivo"
                break
            elif palavra in negativos:
                classificacao = "negativo"
                break
        
        acertou = classificacao == gabarito
    except:
        palavras_previstas = []
        classificacao = "indefinido"
        acertou = False
         
    return {
        "previsoes": palavras_previstas,
        "classificacao": classificacao,
        "gabarito": gabarito,
        "acertou": acertou
    }