import spacy
import pt_core_news_sm

def imprimeEntidadesSpacy( texto: str ) -> str:
    """
    Realiza o processamento do texto de entrada com spaCy, imprime informações de entidades obtidas após o processamento e retorna o texto processado.
    O formato de impressão deve ser o seguinte, para cada token:

        <entidade>: <sigla_da_entidade>
    """
    doc = None
    pipeline = pt_core_news_sm.load()

    doc = pipeline(texto)
    
    for entidade in doc.ents:
        print(f"{entidade.text}: {entidade.label_}")
    
    texto_processado = texto