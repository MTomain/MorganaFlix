import streamlit as st
import os
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import requests
import json

class RandomForestMovieRecommender:
    def __init__(self, data, features_matrix, overview_embeddings, keywords_embeddings, model, scaler):
        """
        Inicializa o sistema de recomendação com os dados pré-processados.
        """
        self.data = data  # Dataset original
        self.features_matrix = features_matrix  # Matriz de features carregada
        self.overview_embeddings = overview_embeddings  # Embeddings de resumo
        self.keywords_embeddings = keywords_embeddings  # Embeddings de palavras-chave
        self.model = model  # Modelo RandomForest carregado
        self.scaler = scaler  # Scaler carregado

    def recommend(self, movie_title, top_n=10):
        """
        Retorna uma lista de recomendações baseada no filme fornecido.
        """
        if movie_title not in self.data['title'].values:
            raise ValueError(f"O título '{movie_title}' não foi encontrado no dataset.")

        # Localizar o índice do filme
        movie_idx = self.data[self.data['title'] == movie_title].index[0]

        # Extrair os embeddings do filme de referência
        movie_features = self.features_matrix[movie_idx].reshape(1, -1)

        # Calcular similaridade com todos os filmes
        sim_scores = cosine_similarity(self.features_matrix, movie_features).flatten()

        # Ordenar as similaridades em ordem decrescente
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Selecionar os índices dos filmes mais similares (excluindo o próprio filme)
        top_indices = [i[0] for i in sim_scores[1:top_n + 1]]

        # Retornar os títulos dos filmes mais similares
        return self.data.iloc[top_indices]

def fetch_poster(path):
    return f"https://image.tmdb.org/t/p/w500/{path}"


# Configurando o Kaggle API Key (use secrets ou armazene de forma segura)
kaggle_json = {
    "username": "caroltomain",
    "key": "1e5ad3a90e7ece998df423885a91b192"
}

# Salvar o kaggle.json temporariamente para autenticação
with open('kaggle.json', 'w') as f:
    json.dump(kaggle_json, f)

os.environ['KAGGLE_CONFIG_DIR'] = os.getcwd()

# Baixar o dataset do Kaggle, caso necessário
if not os.path.exists("TMDB_movie_dataset_v11.csv"):
    os.system("kaggle datasets download -d arcatex/tmdb-movie-dataset -f TMDB_movie_dataset_v11.csv")
    os.system("unzip TMDB_movie_dataset_v11.csv.zip")

# Caminhos dos arquivos
features_matrix_path = 'features_matrix.npz'
overview_embeddings_path = 'overview_embeddings.npy'
keywords_embeddings_path = 'keywords_embeddings.npy'
random_forest_model_path = 'random_forest_model_v1.5.1.pkl'
scaler_path = 'scaler_v1.5.1.pkl'
df = pd.read_csv('TMDB_movie_dataset_v11.csv')

features_matrix = load_npz(features_matrix_path)
overview_embeddings = np.load(overview_embeddings_path)
keywords_embeddings = np.load(keywords_embeddings_path)
random_forest_model = joblib.load(random_forest_model_path)
scaler = joblib.load(scaler_path)

st.set_page_config(layout="wide")

if __name__ == '__main__':
    recommender = RandomForestMovieRecommender(
        data=df,
        features_matrix=features_matrix,
        overview_embeddings=overview_embeddings,
        keywords_embeddings=keywords_embeddings,
        model=random_forest_model,
        scaler=scaler
    )

    st.title("MorganaFlix - Sistema de Recomendação")
    input_seach = st.text_input(
        "Digite um filme que você gosta:", 
        placeholder="Ex.: The Avengers, The Batman..."
    )

    try:
        recommendations_indices = recommender.recommend(input_seach, top_n=10)
        posters = [
            fetch_poster(path) 
            for path in recommendations_indices["poster_path"]
        ]
    except ValueError as e:
        st.error(e)
        st.stop()

    # Dividindo em duas linhas de 5 colunas cada
    for row in range(2):  # Para duas fileiras
        cols = st.columns(5)  # Cada fileira com 5 colunas
        for i in range(5):  # 5 filmes por fileira
            index = row * 5 + i
            if index < len(posters):
                with cols[i]:
                    st.image(posters[index])
                    st.write(f"**{recommendations_indices.iloc[index]['title']}**")
                    st.write(f"Avaliação: {recommendations_indices.iloc[index]['vote_average']}")
