# Importação de bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud



# Função para treinar o modelo de classificação de sentimentos
def treinar_modelo(dados, coluna_texto, coluna_sentimento):
    """
    Treina um modelo de Regressão Logística para prever sentimentos (polaridade).
    
    Parâmetros:
        dados (DataFrame): Dataset contendo os textos e os sentimentos.
        coluna_texto (str): Nome da coluna com os textos.
        coluna_sentimento (str): Nome da coluna com os sentimentos.

    Retorna:
        float: Acurácia do modelo no conjunto de teste.
    """
    # Vetorizar os textos usando o modelo bag-of-words
    vetorizar = CountVectorizer(max_features=100)
    bag_of_words = vetorizar.fit_transform(dados[coluna_texto])

    # Dividir os dados em treino e teste
    treino, teste, classe_treino, classe_teste = train_test_split(
        bag_of_words,
        dados[coluna_sentimento],
        stratify=dados[coluna_sentimento],
        random_state=71
    )

    # Treinar o modelo de Regressão Logística
    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(treino, classe_treino)

    # Retornar a acurácia do modelo
    return regressao_logistica.score(teste, classe_teste)


# Carregar o dataset
avaliacoes = pd.read_csv('b2w.csv')

# Remover colunas desnecessárias
avaliacoes = avaliacoes.drop(
    ['original_index', 'review_text_processed', 'review_text_tokenized', 
     'rating', 'kfold_polarity', 'kfold_rating'], 
    axis=1
)

# Remover valores nulos
avaliacoes.dropna(inplace=True, axis=0)

# Exibir a distribuição das classes de polaridade
print(avaliacoes['polarity'].value_counts())

# Treinar o modelo e exibir a acurácia
acuracia = treinar_modelo(avaliacoes, 'review_text', 'polarity')
print(f"Acurácia do modelo: {acuracia:.2f}")

# Função para criar uma Word Cloud de avaliações negativas
def word_cloud_neg(dados, coluna_texto):
    """
    Gera uma Word Cloud com palavras de avaliações negativas.
    
    Parâmetros:
        dados (DataFrame): Dataset contendo os textos e os sentimentos.
        coluna_texto (str): Nome da coluna com os textos.
    """
    # Filtrar avaliações negativas (polaridade = 0)
    texto_negativo = dados.query('polarity == 0')

    # Unir todas as avaliações em uma única string
    todas_avaliacoes = ' '.join(texto_negativo[coluna_texto])

    # Gerar a Word Cloud
    nuvem_palavras = WordCloud(
        width=800, 
        height=500, 
        max_font_size=110, 
        collocations=False
    ).generate(todas_avaliacoes)

    # Exibir a Word Cloud
    plt.figure(figsize=(10, 7))
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis('off')
    # plt.show()


# Função para criar uma Word Cloud de avaliações positivas
def word_cloud_pos(dados, coluna_texto):
    """
    Gera uma Word Cloud com palavras de avaliações positivas.
    
    Parâmetros:
        dados (DataFrame): Dataset contendo os textos e os sentimentos.
        coluna_texto (str): Nome da coluna com os textos.
    """
    # Filtrar avaliações positivas (polaridade = 1)
    texto_positivo = dados.query('polarity == 1')

    # Unir todas as avaliações em uma única string
    todas_avaliacoes = ' '.join(texto_positivo[coluna_texto])

    # Gerar a Word Cloud
    nuvem_palavras = WordCloud(
        width=800, 
        height=500, 
        max_font_size=110, 
        collocations=False
    ).generate(todas_avaliacoes)

    # Exibir a Word Cloud
    plt.figure(figsize=(10, 7))
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis('off')
    # plt.show()


# Gerar Word Cloud para avaliações negativas
word_cloud_neg(avaliacoes, 'review_text')

# Gerar Word Cloud para avaliações positivas
word_cloud_pos(avaliacoes, 'review_text')

# nltk.download('all')

# corpus = ['Muito bom esse produto', 'Muito ruim esse produto']
# frequencia = nltk.FreqDist(corpus)
# print(frequencia)

from nltk import tokenize

frase = 'Muito bom esse produto'
token_por_espaco = tokenize.WhitespaceTokenizer()
token_frase = token_por_espaco.tokenize(frase)
print(token_frase)

todas_avaliacoes = [texto for texto in avaliacoes.review_text]
todas_palavras = ' '.join(todas_avaliacoes)

token_dataset = token_por_espaco.tokenize(todas_palavras)
frequencia = nltk.FreqDist(token_dataset)

# print(frequencia)

dataframe_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()), "Frequência": list(frequencia.values())})
print(dataframe_frequencia.head())

# As 10 palavras que mais aparecem.
print(dataframe_frequencia.nlargest(columns="Frequência", n=10))

def grafico(dados, coluna_texto, quantidade):
    todas_palavras = ' '.join([texto for texto in dados[coluna_texto]])
    token_frase = token_por_espaco.tokenize(todas_palavras)
    frequencia = nltk.FreqDist(token_frase)

    dataframe_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()), 
                                         "Frequência": list(frequencia.values())})
    
    dataframe_frequencia = dataframe_frequencia.nlargest(columns="Frequência", n=quantidade)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(dataframe_frequencia, 
                 x="Palavra", y="Frequência", color="lightblue")

    ax.set(ylabel="Contagem")
    plt.show()

grafico(avaliacoes, 'review_text', 30)