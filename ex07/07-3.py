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

def classificar_sentimento_por_embeddings(entrada: dict):
    """
    Classifica a polaridade de uma frase com base na similaridade do embedding
    com protótipos positivos e negativos.
    """

    classificacao = None
    sim_pos = None
    sim_neg = None

    frase = entrada.get("frase", "")
    prototipo_positivo = entrada.get("prototipo_positivo", "")
    prototipo_negativo = entrada.get("prototipo_negativo", "")

    extractor = pipeline(
        "feature-extraction",
        model="prajjwal1/bert-tiny",
        tokenizer="prajjwal1/bert-tiny",
        framework="pt"
    )
    
    embedding_frase = np.array(extractor(frase))[0].mean(axis=0).reshape(1, -1)
    embedding_pos = np.array(extractor(prototipo_positivo))[0].mean(axis=0).reshape(1, -1)
    embedding_neg = np.array(extractor(prototipo_negativo))[0].mean(axis=0).reshape(1, -1)
    
    sim_pos = cosine_similarity(embedding_frase, embedding_pos)[0][0]
    sim_neg = cosine_similarity(embedding_frase, embedding_neg)[0][0]
    
    classificacao = "positivo" if sim_pos > sim_neg else "negativo"

    return classificacao, sim_pos, sim_neg