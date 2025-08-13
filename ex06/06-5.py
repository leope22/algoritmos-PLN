import os

class SuppressStderr:
    def __enter__(self):
        self.null_fds = os.open(os.devnull, os.O_RDWR)
        self.old_stderr = os.dup(2)
        os.dup2(self.null_fds, 2)

    def __exit__(self, exc_type, exc_value, traceback):
        os.dup2(self.old_stderr, 2)
        os.close(self.null_fds)
        os.close(self.old_stderr)

# Redireciona stderr (para evitar avisos do Transformers no notebook)
def bloquear_stderr_nativo():
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

bloquear_stderr_nativo()

def avaliaCoberturaNER(entrada: dict) -> float:
    """
    Avalia a cobertura de um modelo NER sobre um corpus pequeno. Considere que o agrupamento de entidades já é corretamente realizado pelo pipeline, ao passar o parâmetro "aggregation_strategy".
    Considere também que podem existir pequenas variações entre as entidades reconhecidas e as entidades esperadas (por exemplo, 'Google' vs 'Google Inc'), mas que ainda seja consideradas válidas.
    Ou seja, para cada entidade detectada (predita), caso ela contenha parte do texto da entidade (seja maiusculo ou minusculo) esperada (real), deve ser consideradas válida/correta.
    
    A entrada é um dicionário com dois campos:
        "corpus": uma lista de strings, onde cada string é uma sentença que será processada pelo modelo NER.

        "anotacoes": uma lista de listas. Cada sublista contém as entidades nomeadas corretas (anotadas manualmente) presentes na sentença correspondente do "corpus".

    A cobertura é a porcentagem de entidades anotadas que foram corretamente encontradas pelo modelo, calculada como:
        cobertura = entidades_encontradas / total_entidades
    """
    
    with SuppressStderr():
        from transformers import pipeline
        ner_pipeline = pipeline(
            task="ner",
            model="gagan3012/bert-tiny-finetuned-ner",
            aggregation_strategy="simple",
            framework='pt'
        )
    
    total_entidades = 0
    encontradas = 0
    
    corpus = entrada.get("corpus", [])
    anotacoes = entrada.get("anotacoes", [])
    
    for sentenca, anotacao_sent in zip(corpus, anotacoes):
        resultados = ner_pipeline(sentenca)
        
        entidades_detectadas = [r['word'].lower() for r in resultados]
        
        for entidade in anotacao_sent:
            total_entidades += 1
            entidade_lower = entidade.lower()
            
            for detectada in entidades_detectadas:
                if entidade_lower in detectada or detectada in entidade_lower:
                    encontradas += 1
                    break
    
    cobertura = encontradas / total_entidades if total_entidades > 0 else 0.0
    
    return cobertura