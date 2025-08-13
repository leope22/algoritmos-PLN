from difflib import SequenceMatcher
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def avaliar_traducao(entrada: dict):
    """
    Avalia automaticamente uma tradução com BLEU e Fuzzy, e decide se a tradução é aceitável com base em limiares.

    Parâmetros:
    entrada: dict com chaves:
        - "referencia": string com a frase original (inglês)
        - "hipotese": string com a frase traduzida automaticamente
        - "limiar_bleu": float (opcional, default=50.0)
        - "limiar_fuzzy": float (opcional, default=70.0)

    Retorno:
    dict com:
        - "referencia": frase de referência
        - "hipotese": frase gerada
        - "BLEU": score BLEU (0-100)
        - "Fuzzy": similaridade difflib (0-100)
        - "aceitavel": True/False se passou ambos os limiares
    """
    referencia = entrada.get("referencia", "")
    hipotese = entrada.get("hipotese", "")
    limiar_bleu = entrada.get("limiar_bleu", 50.0)
    limiar_fuzzy = entrada.get("limiar_fuzzy", 70.0)

    resultado = {
        "referencia": referencia,
        "hipotese": hipotese,
        "BLEU": 0.0,
        "Fuzzy": 0.0,
        "aceitavel": False
    }

    referencia = referencia.lower()
    hipotese = hipotese.lower()
    
    ref_tokens = [referencia.split()]
    hyp_tokens = hipotese.split()
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smoothing)
    resultado["BLEU"] = bleu_score * 100
    
    fuzzy_score = SequenceMatcher(None, referencia, hipotese).ratio()
    resultado["Fuzzy"] = fuzzy_score * 100
    
    resultado["aceitavel"] = (resultado["BLEU"] >= limiar_bleu) and (resultado["Fuzzy"] >= limiar_fuzzy)

    return resultado