from typing import List, Dict
import spacy

def extraiFeatures(texto: str) -> List[Dict]:
    """
    Extrai características avançadas de cada token de uma sentença em português. As características de cada token devem ser agrupadas em dicionários que devem ser retornados juntos.
    As características incluem PoS, lemas, dependências, contexto expandido, NER e relações sintáticas, apresentando as seguintes chaves:

    # Token atual
    'tok',
    'tok_lower',
    'tok_len',
    'tok_is_title',
    'tok_is_digit',
    'tok_is_alpha',

    # PoS, sintaxe e lema
    'pos',
    'tag',
    'lemma',
    'dep',
    'is_root',
    'head_text',
    'head_pos',

    # Entidades nomeadas
    'ner_type',
    'ner_position',

    # Verificações sintáticas
    'is_verb',
    'is_noun',
    'is_adj',

    # Contexto anterior
    'prev_word',
    'prev_lemma',
    'prev_pos',

    # Contexto seguinte
    'next_word',
    'next_lemma',
    'next_pos',
    """

    pipeline = spacy.load("pt_core_news_sm")
    tokens = pipeline(texto)

    MARCACAO_INICIO = "<START>"
    MARCACAO_FIM = "<END>"
    features_sentenca = []

    for i, token in enumerate(tokens):
        features = {
            'tok': token.text,
            'tok_lower': token.text.lower(),
            'tok_len': len(token.text),
            'tok_is_title': token.text.istitle(),
            'tok_is_digit': token.text.isdigit(),
            'tok_is_alpha': token.text.isalpha(),
            
            'pos': token.pos_,
            'tag': token.tag_,
            'lemma': token.lemma_,
            'dep': token.dep_,
            'is_root': token.dep_ == 'ROOT',
            'head_text': token.head.text,
            'head_pos': token.head.pos_,
            
            'ner_type': token.ent_type_ if token.ent_type_ else 'O',
            'ner_position': token.ent_iob_,
            
            'is_verb': token.pos_ == 'VERB',
            'is_noun': token.pos_ == 'NOUN',
            'is_adj': token.pos_ == 'ADJ',
            
            'prev_word': tokens[i-1].text if i > 0 else MARCACAO_INICIO,
            'prev_lemma': tokens[i-1].lemma_ if i > 0 else MARCACAO_INICIO,
            'prev_pos': tokens[i-1].pos_ if i > 0 else MARCACAO_INICIO,
            
            'next_word': tokens[i+1].text if i < len(tokens)-1 else MARCACAO_FIM,
            'next_lemma': tokens[i+1].lemma_ if i < len(tokens)-1 else MARCACAO_FIM,
            'next_pos': tokens[i+1].pos_ if i < len(tokens)-1 else MARCACAO_FIM,
        }
        features_sentenca.append(features)
    
    return features_sentenca