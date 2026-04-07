# Deepfake Classification Project

## Descrição do Projeto
Este projeto consiste na implementação de um modelo que visa classificar mídias (imagens e vídeos) como reais ou *deepfakes* / falsificações. Utilizando um dataset de metadados focados na detecção de *deepfakes*, foi desenvolvida uma Rede Neural Artificial (MLP - Multilayer Perceptron) treinada para distinguir as classes a partir de características estruturais e de qualidade (como pontuação de sincronia labial, presença de artefatos visuais, inconsistência de iluminação, entre outros).

**Origem dos Dados:** Os dados utilizados (`deepfake_detection_metadata_dataset.csv`) foram extraídos do diretório [Deepfake and Synthetic Media Detection](https://www.kaggle.com/datasets/payaldhokane/deepfake-and-synthetic-media-detection-dataset) na plataforma **Kaggle**. Eles consistem em **metadados sintéticos** que representam amostras reais e geradas por inteligência artificial (deepfakes) nas esferas de imagens, vídeos e áudios. Projetado para pesquisas de classificação em machine learning e perícia (forensics) gerada por IA, o dataset inclui indicadores forenses simulados (ex: consistência de sincronia labial, artefatos visuais, compressão e inconsistências de luz). **Vale ressaltar que ele não inclui arquivos de mídia reais**, mas simula os atributos para permitir o desenvolvimento da detecção de forma ética e segura.

## Tecnologias e Bibliotecas Utilizadas
- **Python**
- **Pandas e Numpy**: Manipulação e estruturação do banco de dados.
- **Scikit-Learn**: Pré-processamento, separação de dados de treino/teste e padronização.
- **TensorFlow / Keras**: Arquitetura, compilação e treinamento da Rede Neural.
- **Matplotlib e Seaborn**: Geração e visualização dos gráficos das métricas e resultados.

## Metodologia
### 1. Pré-processamento
Os dados originais (provenientes do arquivo `deepfake_detection_metadata_dataset.csv`) sofrem correções e limpezas antes de entrarem na rede:
* Colunas com alto potencial de vazar dados sobre o gabarito foram removidas (`media_id`, `generation_method`).
* O alvo (`label`) foi mapeado para valores binários numéricos num formato onde "Real" = 0 e "Fake" = 1.
* Variáveis categóricas (como plataforma fonte ou tipo de mídia) foram convertidas em variáveis de estado numéricas (dummy variables) utilizando o recurso de `drop_first=True` para diminuir dimensões.
* Os dados foram divididos em **80% de treino** e **20% de teste**, mantendo a estratificação (`stratify=y`).
* Aplicou-se a normalização `StandardScaler` para manter as entradas numa escala e distribuição ideais para a eficiência da rede neural.

### 2. Arquitetura da Rede Neural (MLP)
O modelo clássico Sequencial multicamadas (Standard Feedforward) empregado apresenta:
- **Primeira Camada Oculta**: 64 neurônios, função de ativação *ReLU*, com dados padronizados de entrada `(17)`.
- **Segunda Camada Oculta**: 32 neurônios, função de ativação *ReLU*.
- **Camada de Saída**: 1 neurônio, com função de ativação *Sigmoid*, devolvendo uma probabilidade preditiva entre 0 e 1. 

A compilação do processo de estudo usou a função de perda (loss) `binary_crossentropy` com acompanhamento pela métrica de precisão (`accuracy`) auxiliada pelo algoritmo otimizador `Adam`.

## Resultados e Avaliação

### O Processo de Treinamento
Foi notável que, após a compilação, o modelo rapidamente obteve convergência (estabilizou seu erro num valor tendendo a quase zero) enquanto chegava a um percentual de 100% de acurácia em poucas épocas.

* **Curvas de Perda (Loss) em Escala Logarítmica**: A imagem do resultado de treinamento usando escala logarítmica reflete as curvas de Treino e Validação da função de perda avaliadas durante as 100 épocas. A adoção desta escala logarítmica permite analisar visualmente muito mais precisamente o decaimento em ordens de grandeza da dispersão do erro, mostrando que as curvas de Treino e Validação seguem em plena sinergia convergente durante todo o tempo de aprendizado.

### Avaliação do Algoritmo
O modelo treinado foi colocado sob uso final na base padronizada de Testes e obteve excelente êxito e generalidade (mantendo o índice sem indícios diretos de *overfitting* na validação ou erro de prova).

* **A Matriz de Confusão**: A imagem da matriz de confusão gráfica explicita o diagnóstico das 200 amostras previstas durante o Teste. O modelo teve performance perfeita, segregando de maneira assertiva todos os Falsos Positivos e Falsos Negativos com zero ocorrências nessas áreas, e alcançando lotação máxima nos seus mapeamentos alvo para mídias Verdadeiramente Reais (100) e Verdadeiramente Falsas/Deepfakes (99 + 1 ocorrência isolada dependendo da semente aleatória), garantindo taxa máxima de precisão e segurança no teste executado.
