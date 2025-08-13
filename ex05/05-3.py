import os
import torch
from transformers import AutoTokenizer, AutoModel

# Redireciona stderr
def bloquear_stderr_nativo():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

bloquear_stderr_nativo()

def avaliar_uas(entrada: dict):
    """
    Simula uma avaliação de UAS entre heads preditos e ouro.
    Usa bert-tiny apenas para gerar embeddings (não faz parsing real).
    """

    uas_total = 0.0
    frases = entrada.get("frases", [])
    heads_ouro = entrada.get("heads_ouro", [])

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    model = AutoModel.from_pretrained("prajjwal1/bert-tiny")

    total_tokens = 0
    acertos = 0

    for i, frase in enumerate(frases):
        inputs = tokenizer(frase, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state[0]
        embs = embeddings[1:-1]

        if embs.size(0) != len(heads_ouro[i]):
            continue

        norm_embs = embs / embs.norm(dim=1, keepdim=True)
        sim_matrix = torch.matmul(norm_embs, norm_embs.T)
        sim_matrix.fill_diagonal_(-float("inf"))

        heads_pred = torch.argmax(sim_matrix, dim=1).tolist()
        heads_gold = heads_ouro[i]

        for pred, real in zip(heads_pred, heads_gold):
            if pred == real:
                acertos += 1

        total_tokens += len(heads_gold)

    if total_tokens > 0:
        uas_total = (acertos / total_tokens)

    return uas_total