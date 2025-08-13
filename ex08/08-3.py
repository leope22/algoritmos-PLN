import os

# Redireciona stderr nativamente
def bloquear_stderr_nativo():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

bloquear_stderr_nativo()

import numpy as np
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

def comparar_impacto_contexto(entrada: dict):
    """
    Calcula a similaridade da embedding de uma palavra em contexto curto vs contexto longo.
    
    Parâmetros:
    entrada: dict com chaves:
        - "palavra": palavra alvo (minúscula, sem acento)
        - "frase_curta": frase com pouco contexto
        - "frase_longa": frase com mais contexto
    
    Retorno:
    dict com:
        - "palavra": palavra alvo
        - "tokens_curta": tokens da frase curta
        - "tokens_longa": tokens da frase longa
        - "indice_curta": posição da palavra na frase curta
        - "indice_longa": posição da palavra na frase longa
        - "similaridade": valor da similaridade de cosseno
        - "impacto_contexto": 1 - similaridade
    """
    modelo = pipeline("feature-extraction", model="prajjwal1/bert-tiny", framework='pt')
    
    palavra = entrada.get("palavra", "").lower()
    frase_curta = entrada.get("frase_curta", "")
    frase_longa = entrada.get("frase_longa", "")
    
    tokens_curta = []
    tokens_longa = []
    idx_curta = -1
    idx_longa = -1
    similaridade = 0.0

    try:
        embeddings_curta = modelo(frase_curta)
        tokens_curta = [token for token in modelo.tokenizer.tokenize(frase_curta)]
        idx_curta = tokens_curta.index(palavra) if palavra in tokens_curta else -1
        
        embeddings_longa = modelo(frase_longa)
        tokens_longa = [token for token in modelo.tokenizer.tokenize(frase_longa)]
        idx_longa = tokens_longa.index(palavra) if palavra in tokens_longa else -1
        
        if idx_curta != -1 and idx_longa != -1:
            emb_curta = np.array(embeddings_curta[0][idx_curta]).reshape(1, -1)
            emb_longa = np.array(embeddings_longa[0][idx_longa]).reshape(1, -1)
            
            similaridade = cosine_similarity(emb_curta, emb_longa)[0][0]
    except:
        pass
    
    return {
        "palavra": palavra,
        "tokens_curta": tokens_curta,
        "tokens_longa": tokens_longa,
        "indice_curta": idx_curta,
        "indice_longa": idx_longa,
        "similaridade": round(similaridade, 4),
        "impacto_contexto": round(1 - similaridade, 4)
    }