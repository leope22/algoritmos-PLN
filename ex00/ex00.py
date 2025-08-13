import numpy as np
import pandas as pd

def criaMatrizes(val_A, val_B):
    """
    Gera duas matrizes com conteudo predefinido
    """
    ExA = np.array(val_A)
    ExB = np.array(val_B)
    return ExA, ExB

def multiplicaMatrizes(ExA, ExB):
    """
    Gera uma matriz atraves da multilicacao de outras duas
    """
    # Para multiplicação de matrizes, o número de colunas da primeira (A)
    # deve ser igual ao número de linhas da segunda (B).
    if ExA.shape[1] != ExB.shape[0]:
        print(f"Erro: As matrizes não podem ser multiplicadas. Dimensões incompatíveis: {ExA.shape} e {ExB.shape}")
        return None
    ExC = np.dot(ExA, ExB)
    return ExC

def mediaStdMatriz(M):
    """
    Calcula a media e desvio padrao das linhas e colunas da matriz M
    """
    media_linhas = np.mean(M, axis=1)
    media_colunas = np.mean(M, axis=0)
    # ddof=1 calcula o desvio padrão da amostra. Se fosse da população, seria ddof=0 (padrão).
    std_linhas = np.std(M, axis=1, ddof=1)
    std_colunas = np.std(M, axis=0, ddof=1)
    return media_linhas, media_colunas, std_linhas, std_colunas

def duasUltimasColunasMedia(M):
    """
    Gera uma matriz com os valores das duas ultimas colunas de M e calcula a média geral
    """
    ExD = M[:, -2:]
    media_D = np.mean(ExD)
    return ExD, media_D

def matrizLinhasColunas(M):
    """
    Gera uma matriz com os valores das linhas e colunas de M com indice 1 e 2
    """
    # Garante que a matriz M seja grande o suficiente para o slice [1:3, 1:3]
    if M.shape[0] < 3 or M.shape[1] < 3:
        print(f"Erro: A matriz de dimensão {M.shape} é pequena demais para extrair o bloco [1:3, 1:3].")
        return None
    ExE = M[1:3, 1:3] 
    return ExE

def matrizZeros(M, N):
    """
    Gera uma matriz com M linhas e N colunas, preenchendo seu conteudo com 0
    """
    ExF = np.zeros((M, N))
    return ExF

def vetorVs(M, V):
    """
    Gera um vetor com M elementos, preenchendo seu conteudo com V
    """
    ExG = np.full(M, V)
    return ExG

def fatorial(n):
    """
    Calcula o fatorial de n (n!)
    """
    if n < 0:
        return "Erro: Fatorial não definido para números negativos"
    if n == 0:
        return 1
    # A implementação com np.prod é inteligente, mas math.factorial é mais comum.
    # Usando a sua implementação:
    fat = np.prod(np.arange(1, n + 1, dtype=np.int64)) # Usar int64 para evitar overflow
    return fat

def pegaElementoPeloNome(df, C_name, I_name):
    """
    Recupera o elemento do DataFrame df com base no nome da coluna e da linha.
    """
    elem = df.loc[I_name, C_name]
    return elem

def pegaElementoPelaPosicao(df, C_pos, I_pos):
    """
    Recupera o elemento do DataFrame df com base na posição da coluna e da linha.
    """
    elem = df.iloc[I_pos, C_pos]
    return elem

def main():
    """
    Função principal que executa e testa todas as outras funções.
    """

    # --- Teste 1: criaMatrizes e multiplicaMatrizes ---
    print("\n[Teste 1: Criação e Multiplicação de Matrizes]")
    lista_A = [[1, 2], [3, 4]]
    lista_B = [[5, 6], [7, 8]]
    matriz_A, matriz_B = criaMatrizes(lista_A, lista_B)
    print(f"Matriz A criada:\n{matriz_A}")
    print(f"Matriz B criada:\n{matriz_B}")
    
    matriz_C = multiplicaMatrizes(matriz_A, matriz_B)
    print(f"Resultado da multiplicação (A * B):\n{matriz_C}")

    # --- Teste 2: mediaStdMatriz ---
    print("\n[Teste 2: Média e Desvio Padrão]")
    matriz_teste = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    print(f"Usando a matriz:\n{matriz_teste}")
    media_l, media_c, std_l, std_c = mediaStdMatriz(matriz_teste)
    print(f"Média das linhas: {media_l}")
    print(f"Média das colunas: {media_c}")
    print(f"Desvio padrão das linhas: {np.round(std_l, 2)}")
    print(f"Desvio padrão das colunas: {np.round(std_c, 2)}")

    # --- Teste 3: duasUltimasColunasMedia ---
    print("\n[Teste 3: Duas Últimas Colunas e Média]")
    matriz_D, media_D = duasUltimasColunasMedia(matriz_teste)
    print(f"Duas últimas colunas da matriz de teste:\n{matriz_D}")
    print(f"Média dos elementos dessas colunas: {media_D:.2f}")

    # --- Teste 4: matrizLinhasColunas ---
    print("\n[Teste 4: Extração de Submatriz]")
    matriz_E = matrizLinhasColunas(matriz_teste)
    print(f"Submatriz extraída (linhas e colunas com índice 1 e 2):\n{matriz_E}")

    # --- Teste 5: matrizZeros ---
    print("\n[Teste 5: Matriz de Zeros]")
    linhas, colunas = 3, 4
    matriz_F = matrizZeros(linhas, colunas)
    print(f"Matriz de zeros {linhas}x{colunas}:\n{matriz_F}")

    # --- Teste 6: vetorVs ---
    print("\n[Teste 6: Vetor com Valor Específico]")
    tamanho, valor = 5, 99
    vetor_G = vetorVs(tamanho, valor)
    print(f"Vetor de tamanho {tamanho} preenchido com {valor}:\n{vetor_G}")

    # --- Teste 7: fatorial ---
    print("\n[Teste 7: Cálculo Fatorial]")
    num = 5
    resultado_fat = fatorial(num)
    print(f"O fatorial de {num} é: {resultado_fat}")
    
    num_grande = 20
    resultado_fat_grande = fatorial(num_grande)
    print(f"O fatorial de {num_grande} é: {resultado_fat_grande}")

    # --- Testes 8 e 9: Funções com DataFrame ---
    print("\n[Testes 8 e 9: Manipulação de DataFrame]")
    # Criando um DataFrame de exemplo
    dados = {'Idade': [25, 30, 22], 'Altura': [1.75, 1.80, 1.65], 'Pontos': [88, 95, 76]}
    indices = ['Ana', 'Bruno', 'Carlos']
    df = pd.DataFrame(dados, index=indices)
    print(f"DataFrame de teste:\n{df}\n")

    # Teste 8: pegaElementoPeloNome
    nome_linha, nome_coluna = 'Bruno', 'Pontos'
    elem_nome = pegaElementoPeloNome(df, nome_coluna, nome_linha)
    print(f"Elemento pego pelo nome (Linha: '{nome_linha}', Coluna: '{nome_coluna}'): {elem_nome}")

    # Teste 9: pegaElementoPelaPosicao
    pos_linha, pos_coluna = 0, 1 # Primeira linha (Ana), Segunda coluna (Altura)
    elem_pos = pegaElementoPelaPosicao(df, pos_coluna, pos_linha)
    print(f"Elemento pego pela posição (Linha: {pos_linha}, Coluna: {pos_coluna}): {elem_pos}")

# Ponto de entrada padrão para scripts Python.
# O código dentro deste 'if' só será executado quando o arquivo for rodado diretamente.
if __name__ == "__main__":
    main()