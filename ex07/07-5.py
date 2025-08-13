import os

# Redireciona stderr nativamente
def bloquear_stderr_nativo():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

bloquear_stderr_nativo()

import numpy as np
from transformers import pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import contextlib

def detectar_fake_news_por_embeddings(entrada: dict):
    """
    Classifica textos como reais ou falsos usando embeddings e KNN.
    Avalia com Precision, Recall e F1 Macro. O dicionário de retorno deve conter as chaves: "precision_macro"], "recall_macro" e "f1_macro"].
    """

    resultados = {}

    textos = entrada.get("textos", [])
    labels = entrada.get("labels", [])  # 0 = real, 1 = fake

    extractor = pipeline(
        "feature-extraction",
        model="prajjwal1/bert-tiny",
        tokenizer="prajjwal1/bert-tiny",
        framework="pt"
    )
    knn = KNeighborsClassifier(n_neighbors=3)
    
    embeddings = []
    for texto in textos:
        embedding = extractor(texto)
        embedding_np = np.array(embedding).squeeze(0).mean(axis=0)
        embeddings.append(embedding_np)
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42
    )
    
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )
    
    resultados["precision_macro"] = precision
    resultados["recall_macro"] = recall
    resultados["f1_macro"] = f1

    return resultados