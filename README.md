# Implementações de Processamento de Linguagem Natural

Este repositório contém uma coleção de funções em Python para realizar diversas tarefas de Processamento de Linguagem Natural (PLN), desde a limpeza e pré-processamento de texto até a implementação de componentes de modelos avançados como Transformers e sistemas de Tradução Automática Neural.

## Estrutura do Repositório

As implementações estão organizadas nas seguintes categorias:

### 1. Pré-processamento e Limpeza de Texto
Funções para normalizar e limpar textos brutos.

- **`normalizaEmails()`**: Normaliza a ocorrência de endereços de e-mail no texto.
- **`removeTagsHTML()`**: Remove qualquer tag HTML do texto.
- **`normalizaNumeros()`**: Normaliza a ocorrência de números no texto.
- **`removeEspacosExtras()`**: Remove espaços em branco adicionais.
- **`removeStopwordsNLTK()`**: Remove stopwords da língua portuguesa usando NLTK.
- **`realizaStemmingNLTK()`**: Reduz palavras à sua raiz (stemming) com NLTK.
- **`preprocessamentoDeTextoNLTK()`**: Orquestra múltiplas etapas de pré-processamento com NLTK.

### 2. Representação de Texto
Técnicas para converter texto em formatos numéricos.

- **`criaRepresentacaoOneHot()`**: Cria representações one-hot para tokens de texto usando scikit-learn.
- **`criaMatrizTermoDocumento()`**: Gera uma matriz de frequência termo-documento com scikit-learn.
- **`criaRepresentacaoTFIDF()`**: Cria representações TF-IDF para o texto com scikit-learn.
- **`criaWordCloud()`**: Gera uma nuvem de palavras e retorna a frequência das 3 palavras mais comuns.
- **`criaMatrizPMI()`**: Gera uma matriz de Pointwise Mutual Information (PMI) com base em coocorrências.

### 3. Embeddings e Similaridade
Funções para trabalhar com word embeddings e calcular similaridade semântica.

- **`similaridadeWord2Vec()`**: Treina um modelo Word2Vec e encontra os termos mais similares a uma palavra-alvo.
- **`criaVetorDocumento()`**: Gera um vetor para um documento calculando a média dos vetores de suas palavras.
- **`documentosSimilares()`**: Encontra os documentos mais similares a um texto de entrada em um corpus.
- **`criaMatrizSimilaridade()`**: Gera embeddings de frases e calcula a matriz de similaridade de cosseno entre elas.
- **`comparaEmbeddingsContextuais()`**: Compara embeddings de uma mesma palavra em diferentes contextos.

### 4. Análise Morfossintática
Funções para extração de características morfológicas e sintáticas.

- **`imprimePOSTaggingSpacy()`**: Imprime informações de Part-of-Speech (POS) Tagging usando spaCy.
- **`imprimeEntidadesSpacy()`**: Imprime informações de Entidades Nomeadas (NER) usando spaCy.
- **`contaPosTags()`**: Conta a frequência de cada POS tag em um conjunto de documentos.
- **`criaHMM()`**: Treina um modelo HMM (Hidden Markov Model) para POS Tagging com NLTK.
- **`calculaProbabilidadeSequencia()`**: Realiza o POS tagging de um texto e retorna a probabilidade da sequência.
- **`extraiFeatures()`**: Extrai um conjunto rico de características de cada token com spaCy (lema, PoS, NER, etc.).

### 5. Análise Sintática e Parsing
Algoritmos para análise da estrutura gramatical das sentenças.

- **`cky_parse()`**: Verifica se uma sentença é aceita por uma gramática livre de contexto usando o algoritmo CKY.
- **`simulate_trasition_parser()`**: Simula um parser de dependências baseado em transições (Shift-Reduce).
- **`avaliar_uas()`**: Simula uma avaliação de Unlabeled Attachment Score (UAS).
- **`validaArvoreDependencia()`**: Verifica se um conjunto de dependências forma uma árvore válida.
- **`extrairDependenciasSpacy()`**: Extrai e representa a árvore de dependência de uma sentença com spaCy.
- **`extrairSentencasPorEstrutura()`**: Extrai sentenças do corpus Treebank (NLTK) que correspondem a uma estrutura sintática.

### 6. Semântica e Análise de Discurso
Implementações focadas na interpretação do significado e do discurso.

- **`desambiguar_wsd()`**: Implementa uma estratégia de Desambiguação de Sentido de Palavra (WSD) com embeddings.
- **`avaliarWSD()`**: Avalia um sistema WSD calculando a similaridade entre sentenças com palavras ambíguas.
- **`detectar_papeis_semanticos()`**: Implementa uma estratégia simplificada de Rotulagem de Papéis Semânticos (SRL).
- **`comparaVozAtivaVozPassiva()`**: Compara embeddings para analisar a representação de papéis semânticos.
- **`avaliaCoberturaNER()`**: Avalia a cobertura de um modelo de Reconhecimento de Entidades Nomeadas (NER).

### 7. Classificação e Análise de Sentimento
Modelos para classificação de texto, como análise de sentimento e detecção de fake news.

- **`treinaNaiveBayes()`**: Realiza o treinamento manual de um classificador Naive Bayes.
- **`classificadorUsandoLexicos()`**: Classifica um texto usando os léxicos da Wordnet Affect BR.
- **`classificar_sentimento_por_embeddings()`**: Classifica a polaridade de sentenças via similaridade de embeddings.
- **`classificaTweets()`**: Realiza inferência e calcula a acurácia em uma base de dados de discurso de ódio.
- **`detectar_fake_news_por_embeddings()`**: Detecta notícias falsas com embeddings e um classificador KNN.
- **`simular_propagacao_fake_news()`**: Simula a propagação de notícias em uma rede social (grafo).

### 8. Modelos de Linguagem e Transformers
Funções para explorar e utilizar modelos baseados na arquitetura Transformer.

- **`prever_mascara()`**: Utiliza um Masked Language Model (MLM) para prever palavras ausentes.
- **`classificar_com_prompt_mlm()`**: Realiza classificação de sentimento usando um MLM e templates (prompts).
- **`comparar_impacto_contexto()`**: Mede como o contexto influencia a embedding de uma palavra em um Transformer.

### 9. Tradução Automática Neural (NMT)
Implementação de componentes essenciais para um sistema de tradução automática.

- **`traduzir_palavra_a_palavra()`**: Mapeia embeddings entre dois idiomas usando análise de componentes principais.
- **`avaliar_traducao()`**: Avalia a qualidade de uma tradução baseada em embeddings.
- **`classeVocabulary()`**: Implementa a classe de vocabulário para um modelo NMT.
- **`classeNMTEncoder()`**: Implementa a arquitetura do Encoder para um modelo NMT.
- **`classeNMTDecoder()`**: Implementa a arquitetura do Decoder para um modelo NMT.
