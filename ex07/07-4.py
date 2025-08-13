import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from auxiliar import preprocessamentoDeTextoNLTK

from datasets import load_dataset
from datasets import disable_progress_bar

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

try:
    nltk.data.find('vader_lexicon')

except LookupError:
    nltk.download('vader_lexicon', quiet=True)

disable_progress_bar()

def classificaTweets( n: int ):
    """
    Classifica um subconjunto de tweets de um dataset Hugging Face e calcula a acuracia. Para cada tweet, verificar o valor de "compound", retornado pelo analisador NLTK:
    Se compound >= 0.05 -> sentimento positivo, não há discurso de ódio
    Se compound <= -0.05 -> sentimento negativo, há discurso de ódio
    Caso contrário -> sentimento neutro, não há discurso de ódio
    """
    
    dataset = load_dataset("tweets-hate-speech-detection/tweets_hate_speech_detection", split='train', trust_remote_code=True)
    dataset = dataset.select(range(n))
    analisador = SentimentIntensityAnalyzer()
    
    labels_preditos, labels_reais = [], []
    acc = None

    for tweet in dataset:
        texto = tweet['tweet']
        texto_processado = preprocessamentoDeTextoNLTK(texto)
        
        scores = analisador.polarity_scores(texto_processado)
        compound = scores['compound']
        
        if compound >= 0.05:
            pred = 0
        elif compound <= -0.05:
            pred = 1
        else:
            pred = 0
        
        true_label = tweet['label']
        
        labels_preditos.append(pred)
        labels_reais.append(true_label)
    
    acc = accuracy_score(labels_reais, labels_preditos)

    return acc