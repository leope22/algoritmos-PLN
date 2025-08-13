import pandas as pd

from auxiliar import preprocessamentoDeTextoNLTK, removeAcentos
from sklearn.metrics import classification_report

def classificadorUsandoLexicos( entrada: list[tuple] ):
    """
    Função que recebe uma lista de textos (string) e suas respectivas classes ("positivo", "negativo" ou "neutro")..
    A classificação deve ser feita com base na contagem de palavras presentes em léxicos de sentimento. Os léxicos são extraídos do arquivo wordnetaffectbr_valencia.csv, que contém uma lista de palavras associadas à valência emocional positiva (+) ou negativa (-).
    A função deve, além de predizer a classe de cada texto, calcular métricas de classificação - use o método "classification_report" da Scikit-Learn, com o parâmetro "zero_division=0.0" para evitar erros.
    """
    
    lexicos = pd.read_csv('wordnetaffectbr_valencia.csv', encoding='latin1', sep=';')
    lexicos['Wordnet Affect BR'] = lexicos['Wordnet Affect BR'].apply(removeAcentos)

    lexicos_positivos = lexicos[lexicos['Valência'] == '+']['Wordnet Affect BR'].to_list()
    lexicos_negativos = lexicos[lexicos['Valência'] == '-']['Wordnet Affect BR'].to_list()
    
    classes_reais, classes_preditas = [], []
    report = None
    
    for texto, classe_real in entrada:
        texto_processado = preprocessamentoDeTextoNLTK(texto)
        texto_processado = removeAcentos(texto_processado)
        
        count_pos = sum(1 for palavra in texto_processado.split() if palavra in lexicos_positivos)
        count_neg = sum(1 for palavra in texto_processado.split() if palavra in lexicos_negativos)
        
        if count_pos > count_neg:
            classe_predita = 'positivo'
        elif count_neg > count_pos:
            classe_predita = 'negativo'
        else:
            classe_predita = 'neutro'
            
        classes_reais.append(classe_real)
        classes_preditas.append(classe_predita)
    
    report = classification_report(classes_reais, classes_preditas, zero_division=0.0)
    
    return report