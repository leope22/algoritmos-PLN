import os

# Redireciona stderr nativamente
def bloquear_stderr_nativo():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

bloquear_stderr_nativo()

import numpy as np
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import contextlib

def detectar_papeis_semanticos(entrada: dict):
    """
    Simula a detecção de papéis semânticos pela distância vetorial ao verbo.
    Filtra tokens não alfabéticos (como pontuações).
    """

    papeis = {}

    frase = entrada.get("frase", "")
    verbo = entrada.get("verbo", "").lower()

    extractor = pipeline(
        "feature-extraction",
        model="prajjwal1/bert-tiny",
        tokenizer="prajjwal1/bert-tiny",
        framework="pt"
    )

    features = extractor(frase)
    
    tokens = extractor.tokenizer.tokenize(frase)
    tokens = [token.replace("##", "") for token in tokens if token.replace("##", "").isalpha()]
    
    try:
        verbo_idx = tokens.index(verbo)
    except ValueError:
        return papeis
    
    verbo_vector = np.array(features[0][verbo_idx]).reshape(1, -1)
    
    similarities = []
    for i, token in enumerate(tokens):
        if i != verbo_idx:
            token_vector = np.array(features[0][i]).reshape(1, -1)
            similarity = cosine_similarity(verbo_vector, token_vector)[0][0]
            similarities.append((token, similarity))
    
    if not similarities:
        return papeis
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    if similarities:
        papeis["mais_proximo"] = similarities[0][0]
        papeis["mais_distante"] = similarities[-1][0]

    return papeis