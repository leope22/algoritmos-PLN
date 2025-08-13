import random

def criar_grafo_simples(num_nos: int, conexoes_por_no: int = 2) -> dict:
    """
    Gera um grafo simples como dicionário de adjacências.
    
    Parâmetros:
    - num_nos: int, número total de nós no grafo.
    - conexoes_por_no: int, número de conexões por nó.
    
    Retorno:
    - dict {nó: set(vizinhos)}
    """
    grafo = {n: set() for n in range(num_nos)}
    
    for no in grafo:
        while len(grafo[no]) < conexoes_por_no:
            alvo = random.randint(0, num_nos - 1)
            if alvo != no:
                grafo[no].add(alvo)
                grafo[alvo].add(no)  # conexão bidirecional
    
    return grafo

def simular_propagacao_fake_news(entrada: dict):
    """
    Simula a propagação de uma notícia (real ou fake) em um grafo simples, de forma determinística.
    
    Parâmetros:
    entrada: dict com chaves:
        - "seed": int, valor para fixar a aleatoriedade
        - "num_nos": int, número de nós no grafo
        - "conexoes_por_no": int, conexões por nó
        - "origem": int, nó de origem
        - "tipo": str, 'real' ou 'fake'
        - "prob_share_real": float, probabilidade de compartilhar notícia real
        - "prob_share_fake": float, probabilidade de compartilhar notícia fake

    Retorno:
    estado: dict {nó: True/False}, quem compartilhou.
    profundidade: dict {nó: profundidade de propagação}.
    grafo: dict {nó: set(vizi{nhos)}, estrutura do grafo.
    configuracao_grafo: dict {"origem" e "tipo_noticia"}
    """
    seed = entrada.get("seed", 42)
    random.seed(seed)  # Fixando aleatoriedade

    num_nos = entrada.get("num_nos", 10)
    conexoes_por_no = entrada.get("conexoes_por_no", 2)
    origem = random.randint(0, num_nos - 1)
    tipo = entrada.get("tipo", "fake")
    prob_share_real = entrada.get("prob_share_real", 0.2)
    prob_share_fake = entrada.get("prob_share_fake", 0.7)

    # Criar grafo dentro da função
    grafo = criar_grafo_simples(num_nos, conexoes_por_no)
    configuracao_grafo = {"origem": origem, "tipo_noticia": tipo}
    estado = None
    profundidade = None

    estado = {no: False for no in grafo}
    profundidade = {no: None for no in grafo}
    
    estado[origem] = True
    profundidade[origem] = 0
    
    prob_share = prob_share_fake if tipo == "fake" else prob_share_real
    
    queue = [(origem, 0)]
    visited = set([origem])
    
    while queue:
        current_node, current_depth = queue.pop(0)
        
        if current_node == origem or random.random() <= prob_share:
            estado[current_node] = True
            profundidade[current_node] = current_depth
            
            for neighbor in grafo[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, current_depth + 1))
        else:
            estado[current_node] = False

    return estado, profundidade, grafo, configuracao_grafo