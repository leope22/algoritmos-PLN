import os
import sys

# Redireciona stderr de forma nativa antes de qualquer import
def bloquear_stderr_nativo():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

bloquear_stderr_nativo()

import numpy as np
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

def criaMatrizSimilaridade(frases: list[str]):
    
    embeddings = []
    matriz_similaridade = None
    extractor = pipeline(
        "feature-extraction",
        model="prajjwal1/bert-tiny",
        tokenizer="prajjwal1/bert-tiny",
        framework="pt"
    )

    for frase in frases:
        embedding = extractor(frase)
        embedding_array = np.array(embedding).squeeze(0)
        sentence_embedding = np.mean(embedding_array, axis=0)
        embeddings.append(sentence_embedding)
    
    embeddings_array = np.array(embeddings)
    
    matriz_similaridade = cosine_similarity(embeddings_array)

    return np.round(matriz_similaridade, 2)