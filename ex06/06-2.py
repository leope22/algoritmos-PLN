from transformers import AutoTokenizer, AutoModel
import torch, os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Redireciona stderr
def bloquear_stderr_nativo():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

bloquear_stderr_nativo()

def avaliarWSD( sentencas: list[str] ) -> np.ndarray:
    """
    Função para avaliar manualmente se embeddings de Transformers conseguem distinguir diferentes sentidos de uma palavra ambígua.
    Para obter os embeddings das sentenças, use o vetor correspondente ao token [CLS] que representa o embedding global da sentença.
    Como você não está treinando o modelo, ative o modo de inferência desabilitando o cálculo dos gradientes.
    """

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
    
    matriz_similaridades = None
    
    model.eval()
    
    with torch.no_grad():
        inputs = tokenizer(sentencas, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)

        cls_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        matriz_similaridades = cosine_similarity(cls_embeddings)
        
    return matriz_similaridades