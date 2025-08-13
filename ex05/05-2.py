def simulate_transition_parser(entrada: dict):
    """
    Simula um parser baseado em transições com SHIFT e REDUCE.
    Recebe um dicionário com 'tokens' e 'acoes' como entrada.
    Retorna a árvore final construída ou None se inválido.
    """

    arvore = None
    tokens = entrada.get("tokens", [])
    acoes = entrada.get("acoes", [])

    pilha = []
    token_ptr = 0

    for acao in acoes:
        if acao == "SHIFT":
            if token_ptr >= len(tokens):
                return None
            pilha.append(tokens[token_ptr])
            token_ptr += 1
        elif acao == "REDUCE":
            if len(pilha) < 2:
                return None
            right = pilha.pop()
            left = pilha.pop()
            subarvore = f"({left} {right})"
            pilha.append(subarvore)
    
    if token_ptr == len(tokens) and len(pilha) == 1:
        arvore = pilha[0]

    return arvore