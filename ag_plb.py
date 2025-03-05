import streamlit as st
import pandas as pd
import numpy as np
import random
import folium
from deap import base, creator, tools, algorithms
from streamlit_folium import folium_static

# Função para carregar os dados do CSV
def carregar_dados(uploaded_file):
    if uploaded_file is not None:
        # Lê o arquivo CSV diretamente do upload
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.warning("Por favor, faça o upload de um arquivo CSV.")
        return None

# Interface no Streamlit
st.title("Otimização de Entregas em Goiânia")
st.write("Este aplicativo encontra a melhor rota para entregas em Goiânia usando um algoritmo genético.")

# Exibir a logo da PUC Goiás
st.image("logo.jpg", width=200)  # Certifique-se de que o arquivo da logo está no mesmo diretório

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Carregar o arquivo CSV com os endereços", type=["csv"])

# Carregar os dados se o arquivo foi enviado
df = carregar_dados(uploaded_file)

if df is not None:
    # Coordenadas da Praça Cívica (ponto central de Goiânia)
    latitude_central = -16.6799
    longitude_central = -49.255

    # Função para calcular a distância euclidiana entre dois pontos
    def calcular_distancia(cidade1, cidade2):
        coord1 = np.array([cidade1["Latitude"], cidade1["Longitude"]])
        coord2 = np.array([cidade2["Latitude"], cidade2["Longitude"]])
        return np.linalg.norm(coord1 - coord2)

    # Função de aptidão (distância total da rota)
    def calcular_aptidao(rota):
        distancia_total = 0
        for i in range(len(rota) - 1):
            distancia_total += calcular_distancia(df.iloc[rota[i]], df.iloc[rota[i + 1]])
        distancia_total += calcular_distancia(df.iloc[rota[-1]], df.iloc[rota[0]])  # Retorna ao início
        return (distancia_total,)

    # Configuração do Algoritmo Genético
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individuo", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(df)), len(df))
    toolbox.register("individuo", tools.initIterate, creator.Individuo, toolbox.indices)
    toolbox.register("populacao", tools.initRepeat, list, toolbox.individuo)

    toolbox.register("evaluate", calcular_aptidao)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def otimizar_rota():
        populacao = toolbox.populacao(n=100)
        algorithms.eaSimple(populacao, toolbox, cxpb=0.7, mutpb=0.2, ngen=200, verbose=False)
        melhor_individuo = tools.selBest(populacao, k=1)[0]
        return melhor_individuo

    # Coordenadas médias
    lat_media = df["Latitude"].mean()
    lon_media = df["Longitude"].mean()
    st.write(f"Coordenadas médias das entregas: Latitude {lat_media}, Longitude {lon_media}")

    # Botão para otimizar a rota
    if st.button("Encontrar Melhor Rota"):
        melhor_rota = otimizar_rota()
        st.write("Melhor rota encontrada:", melhor_rota)

        # Criar mapa interativo com OpenStreetMap
        mapa = folium.Map(location=[latitude_central, longitude_central], zoom_start=12, tiles="OpenStreetMap")

        # Adicionar marcador para o ponto de partida
        ponto_inicio = df.iloc[melhor_rota[0]]
        folium.Marker(
            [ponto_inicio["Latitude"], ponto_inicio["Longitude"]],
            popup="Endereço 1 (Ponto de Partida)",
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(mapa)

        # Adicionar marcadores para as entregas intermediárias e incluir informação do próximo destino
        for idx, entrega in enumerate(melhor_rota):
            if idx == 0:
                continue  # O ponto de partida já foi adicionado
            lat, lon = df.iloc[entrega][["Latitude", "Longitude"]]

            # Informar o próximo destino, se não for o último ponto
            if idx < len(melhor_rota) - 1:
                prox_entrega = melhor_rota[idx + 1]
                prox_info = f"Próximo: de {idx + 1} para {idx + 2}"
            else:
                prox_info = "Próximo: Retorno ao Ponto de Partida"

            # Adicionar popup com informações
            popup_info = f"Endereço {idx + 1}\n{prox_info}"
            folium.Marker(
                [lat, lon],
                popup=popup_info,
                icon=folium.Icon(color="blue")
            ).add_to(mapa)

        # Adicionar linhas da rota com início destacado
        for i in range(len(melhor_rota) - 1):
            lat1, lon1 = df.iloc[melhor_rota[i]][["Latitude", "Longitude"]]
            lat2, lon2 = df.iloc[melhor_rota[i + 1]][["Latitude", "Longitude"]]
            folium.PolyLine([(lat1, lon1), (lat2, lon2)], color="red", weight=2.5).add_to(mapa)

        # Destacar o retorno ao ponto de partida
        ultimo_lat, ultimo_lon = df.iloc[melhor_rota[-1]][["Latitude", "Longitude"]]
        folium.PolyLine([(ultimo_lat, ultimo_lon), (ponto_inicio["Latitude"], ponto_inicio["Longitude"])],
                        color="orange", weight=2.5, dash_array='5, 5').add_to(mapa)

        # Exibir marcador para o ponto de chegada
        folium.Marker(
            [ultimo_lat, ultimo_lon],
            popup="Endereço {} (Ponto de Chegada)".format(len(melhor_rota)),
            icon=folium.Icon(color="red", icon="flag")
        ).add_to(mapa)

        # Exibir mapa
        folium_static(mapa)
