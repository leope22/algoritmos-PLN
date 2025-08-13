from collections import defaultdict, deque

def validaArvoreDependencia( arestas ):
    """
    Determina se um conjunto de dependências sintáticas forma uma árvore de dependência válida.
    Uma árvore de dependência válida deve satisfazer as seguintes propriedades:
    (i) Deve haver exatamente uma raiz (ROOT) com grau de entrada 0;
    (ii) A estrutura deve ser conexa (todos os nós acessíveis a partir da raiz); e
    (iii) A estrutura deve ser acíclica (nenhum ciclo entre as dependências).
    
    Pseudo-codigo:

    Para cada (cabeca, dependente) em arestas:
        Adicionar dependente na lista de adjacencia de cabeca
        Incrementar grau de entrada de dependente
        Adicionar cabeca e dependente no conjunto de nos

    Encontrar nos com grau de entrada zero
    Se houver mais de uma raiz ou a raiz nao for 'ROOT':
        Retornar False

    Criar fila com 'ROOT'
    Criar conjunto de visitados

    Enquanto fila nao estiver vazia:
        Remover um no da fila
        Se ja foi visitado, retornar False
        Marcar como visitado
        Adicionar vizinhos na fila

    Retornar True se todos os nos foram visitados
    """
    
    grafo = defaultdict(list)
    grau_entrada = defaultdict(int)
    nos = set()
    arvore_aceita = None
    
    for cabeca, dependente in arestas:
        grafo[cabeca].append(dependente)
        grau_entrada[dependente] += 1
        nos.add(cabeca)
        nos.add(dependente)

    raizes = [n for n in nos if grau_entrada[n] == 0]

    if len(raizes) != 1 or raizes[0] != 'ROOT':
        arvore_aceita = False
    else:
        visitados = set()
        fila = deque(['ROOT'])

        while fila:
            no = fila.popleft()
            if no in visitados:
                arvore_aceita = False
                break
            visitados.add(no)
            for vizinho in grafo[no]:
                fila.append(vizinho)

        if len(visitados) == len(nos):
            arvore_aceita = True
        else:
            arvore_aceita = False

    return arvore_aceita