import pandas as pd
import numpy as np
import random
import folium
from deap import base, creator, tools, algorithms
from streamlit_folium import folium_static
import streamlit as st
import time
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import io
import xlsxwriter
# Função para carregar os dados do CSV
def carregar_dados(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    else:
        st.warning("Por favor, faça o upload de um arquivo CSV.")
        return None

# Função de Haversine para calcular distâncias geográficas
def haversine(coord1, coord2):
    R = 6371.0  # Raio da Terra em quilômetros
    lat1, lon1 = radians(coord1[0]), radians(coord1[1])
    lat2, lon2 = radians(coord2[0]), radians(coord2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distancia = R * c
    return distancia

# Função para calcular a distância entre dois pontos
def calcular_distancia(cidade1, cidade2):
    coord1 = (cidade1["Latitude"], cidade1["Longitude"])
    coord2 = (cidade2["Latitude"], cidade2["Longitude"])
    return haversine(coord1, coord2)

# Função de aptidão (distância total da rota)
def calcular_aptidao(rota):
    distancia_total = 0
    for i in range(len(rota) - 1):
        distancia_total += calcular_distancia(df.iloc[rota[i]], df.iloc[rota[i + 1]])
    distancia_total += calcular_distancia(df.iloc[rota[-1]], df.iloc[rota[0]])  # Retorna ao início
    return (distancia_total,)

# Interface no Streamlit
# Centralizando a primeira imagem com HTML/CSS
st.markdown(
    """
    <div style="display: flex; justify-content: center;">
        <img src="https://via.placeholder.com/200" width="200">
    </div>
    """,
    unsafe_allow_html=True
)

# Criando colunas vazias para centralizar a imagem da PUC Goiás
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.image("logo.jpg", width=200)

st.title("Otimização de Entregas em Goiânia")
st.write("Este aplicativo encontra a melhor rota para entregas em Goiânia usando um algoritmo genético.")

# Upload do arquivo CSV
uploaded_file = st.file_uploader("Carregar o arquivo CSV com os endereços", type=["csv"])

# Carregar os dados se o arquivo foi enviado
df = carregar_dados(uploaded_file)

if df is not None:
    # Coordenadas da Praça Cívica (ponto central de Goiânia)
    latitude_central = -16.6799
    longitude_central = -49.255

    # Coordenadas médias
    lat_media = df["Latitude"].mean()
    lon_media = df["Longitude"].mean()
    st.write(f"Coordenadas médias das entregas: Latitude {lat_media}, Longitude {lon_media}")

    # Configuração do Algoritmo Genético (só ocorre se df estiver definido)
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

    # Função para otimizar a rota com barra de progresso
    def otimizar_rota(cxpb, mutpb, ngen, progress_bar):
        populacao = toolbox.populacao(n=100)
        for gen in range(ngen):
            algorithms.eaSimple(populacao, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=1, verbose=False)
            progress_bar.progress((gen + 1) / ngen)
        melhor_individuo = tools.selBest(populacao, k=1)[0]
        return melhor_individuo

    # Controles deslizantes para definir parâmetros do AG
    st.sidebar.header("Configuração do Algoritmo Genético")
    cxpb = st.sidebar.slider("Taxa de Crossover (cxpb)", 0.1, 1.0, 0.7)
    mutpb = st.sidebar.slider("Taxa de Mutação (mutpb)", 0.01, 1.0, 0.2)
    ngen = st.sidebar.slider("Número de Gerações (ngen)", 10, 1000, 200)

    # Botão para otimizar a rota
    if st.button("Encontrar Melhor Rota"):
        # Barra de progresso para a rota padrão
        st.write("Calculando rota padrão...")
        progress_bar_padrao = st.progress(0)
        start_time = time.time()
        melhor_rota_padrao = otimizar_rota(0.7, 0.2, 200, progress_bar_padrao)
        distancia_padrao = calcular_aptidao(melhor_rota_padrao)[0]
        tempo_padrao = time.time() - start_time

        # Barra de progresso para a rota customizada
        st.write("Calculando rota customizada...")
        progress_bar_customizada = st.progress(0)
        start_time = time.time()
        melhor_rota_customizada = otimizar_rota(cxpb, mutpb, ngen, progress_bar_customizada)
        distancia_customizada = calcular_aptidao(melhor_rota_customizada)[0]
        tempo_customizado = time.time() - start_time

        # Exibir resultados
        dist_percent = abs((distancia_padrao - distancia_customizada)/(distancia_padrao))
        tempo = abs(tempo_padrao - tempo_padrao )
        tempo_percent = (tempo/tempo_padrao)

        st.write(f"Distância total (padrão): {abs(distancia_padrao):.2f} km")
        st.write(f"Distância total (customizada): {distancia_customizada:.2f} km")
        st.write(f"Tempo de execução (padrão): {tempo_padrao:.2f} segundos")
        st.write(f"Tempo de execução (customizada): {tempo_customizado:.2f} segundos")
        st.write(f"Diferença percentual da distância: {dist_percent:.2%}")
        st.write(f"Diferença percentual do tempo: {tempo_percent:.2%}")
        # Análise estatística da diferença
        diferenca = distancia_customizada - distancia_padrao
        st.write(f"Diferença na distância: {diferenca:.2f} km")

        st.write("Observação: As distâncias são calculadas usando a fórmula de Haversine.")

        # Criar mapa interativo para a rota padrão
        mapa_padrao = folium.Map(location=[latitude_central, longitude_central], zoom_start=12, tiles="OpenStreetMap")

        # Adicionar marcadores para o ponto de partida da rota padrão
        ponto_inicio_padrao = df.iloc[melhor_rota_padrao[0]]
        folium.Marker(
            [ponto_inicio_padrao["Latitude"], ponto_inicio_padrao["Longitude"]],
            popup="Endereço 1 (Ponto de Partida)",
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(mapa_padrao)

        # Adicionar marcadores para as entregas intermediárias da rota padrão
        for idx, entrega in enumerate(melhor_rota_padrao):
            if idx == 0:
                continue  # O ponto de partida já foi adicionado
            lat, lon = df.iloc[entrega][["Latitude", "Longitude"]]
            folium.Marker(
                [lat, lon],
                popup=f"Endereço {idx + 1}",
                icon=folium.Icon(color="blue")
            ).add_to(mapa_padrao)

        # Adicionar linhas da rota padrão
        for i in range(len(melhor_rota_padrao) - 1):
            lat1, lon1 = df.iloc[melhor_rota_padrao[i]][["Latitude", "Longitude"]]
            lat2, lon2 = df.iloc[melhor_rota_padrao[i + 1]][["Latitude", "Longitude"]]
            folium.PolyLine([(lat1, lon1), (lat2, lon2)], color="red", weight=2.5).add_to(mapa_padrao)

        # Destacar o retorno ao ponto de partida da rota padrão
        ultimo_lat, ultimo_lon = df.iloc[melhor_rota_padrao[-1]][["Latitude", "Longitude"]]
        folium.PolyLine([(ultimo_lat, ultimo_lon), (ponto_inicio_padrao["Latitude"], ponto_inicio_padrao["Longitude"])],
                        color="orange", weight=2.5, dash_array='5, 5').add_to(mapa_padrao)

        # Exibir mapa padrão
        st.subheader("Mapa com configurações de AG padrão")
        folium_static(mapa_padrao)

        # Criar mapa interativo para a rota customizada
        mapa_customizada = folium.Map(location=[latitude_central, longitude_central], zoom_start=12, tiles="OpenStreetMap")

        # Adicionar marcadores para o ponto de partida da rota customizada
        ponto_inicio_customizado = df.iloc[melhor_rota_customizada[0]]
        folium.Marker(
            [ponto_inicio_customizado["Latitude"], ponto_inicio_customizado["Longitude"]],
            popup="Endereço 1 (Ponto de Partida)",
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(mapa_customizada)

        # Adicionar marcadores para as entregas intermediárias da rota customizada
        for idx, entrega in enumerate(melhor_rota_customizada):
            if idx == 0:
                continue  # O ponto de partida já foi adicionado
            lat, lon = df.iloc[entrega][["Latitude", "Longitude"]]
            folium.Marker(
                [lat, lon],
                popup=f"Endereço {idx + 1}",
                icon=folium.Icon(color="blue")
            ).add_to(mapa_customizada)

        # Adicionar linhas da rota customizada
        for i in range(len(melhor_rota_customizada) - 1):
            lat1, lon1 = df.iloc[melhor_rota_customizada[i]][["Latitude", "Longitude"]]
            lat2, lon2 = df.iloc[melhor_rota_customizada[i + 1]][["Latitude", "Longitude"]]
            folium.PolyLine([(lat1, lon1), (lat2, lon2)], color="red", weight=2.5).add_to(mapa_customizada)

        # Destacar o retorno ao ponto de partida da rota customizada
        ultimo_lat, ultimo_lon = df.iloc[melhor_rota_customizada[-1]][["Latitude", "Longitude"]]
        folium.PolyLine([(ultimo_lat, ultimo_lon), (ponto_inicio_customizado["Latitude"], ponto_inicio_customizado["Longitude"])],
                        color="orange", weight=2.5, dash_array='5, 5').add_to(mapa_customizada)

        # Exibir mapa customizado
        st.subheader("Mapa Customizado")
        folium_static(mapa_customizada)

    # Botão para análise estatística
    if st.button("Análise Estatística"):
        st.write("Executando análise estatística...")

        # Análise 1: 10 rodadas com configuração padrão
        resultados_padrao = []
        for _ in range(10):
            with st.spinner(f"Rodada {_+1}/10 (configuração padrão)..."):
                start_time = time.time()
                melhor_rota = otimizar_rota(0.7, 0.2, 200, st.progress(0))
                distancia = calcular_aptidao(melhor_rota)[0]
                tempo_execucao = time.time() - start_time
                resultados_padrao.append((distancia, tempo_execucao))

        # Criar dataframe com os resultados
        df_padrao = pd.DataFrame(resultados_padrao, columns=["Distância (km)", "Tempo (s)"])
        st.write("Resultados das 10 rodadas com configuração padrão:")
        st.write(df_padrao)

        # Calcular desvio padrão
        distancia_media = df_padrao["Distância (km)"].mean()
        desvio_padrao_distancia = df_padrao["Distância (km)"].std()
        # Calcular o desvio padrão percentual em relação à média
        desvio_padrao_percentual = (desvio_padrao_distancia / desvio_padrao_distancia) * 100
        tempo_medio =  df_padrao["Tempo (s)"].mean()
        desvio_padrao_tempo = df_padrao["Tempo (s)"].std()
        desvio_padrao_percentual_temp = (desvio_padrao_tempo/tempo_medio)*100
        st.write(f"Desvio padrão da distância: {desvio_padrao_distancia:.2f} km")
        st.write(f"Desvio padrão do tempo: {desvio_padrao_tempo:.2f} segundos")
        st.write(f"Desvio padrão percentual: {desvio_padrao_percentual:.2f}%")
        st.write(f"Desvio padrão percentual em tempo: {desvio_padrao_percentual_temp:.2f}%")

        # Gerar histograma
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].hist(df_padrao["Distância (km)"], bins=5, color='blue', alpha=0.7)
        ax[0].set_title("Distribuição das Distâncias")
        ax[0].set_xlabel("Distância (km)")
        ax[0].set_ylabel("Frequência")

        ax[1].hist(df_padrao["Tempo (s)"], bins=5, color='green', alpha=0.7)
        ax[1].set_title("Distribuição dos Tempos")
        ax[1].set_xlabel("Tempo (s)")
        ax[1].set_ylabel("Frequência")

        st.pyplot(fig)

        # # Disponibilizar dataframe para download
        # csv_padrao = df_padrao.to_csv(index=False, sep="|")
        # st.download_button(
        #     label="Download dos resultados padrão",
        #     data=csv_padrao,
        #     file_name="resultados_padrao.csv",
        #     mime="text/csv"
        # )

        # Análise 2: 10 rodadas com configuração customizada e parâmetros aleatórios
        resultados_customizados = []
        for _ in range(10):
            cxpb_rand = random.uniform(0.1, 1.0)
            mutpb_rand = random.uniform(0.01, 0.5)
            ngen_rand = random.randint(50, 500)
            start_time = time.time()
            melhor_rota = otimizar_rota(cxpb_rand, mutpb_rand, ngen_rand, st.progress(0))
            distancia = calcular_aptidao(melhor_rota)[0]
            tempo_execucao = time.time() - start_time
            resultados_customizados.append((cxpb_rand, mutpb_rand, ngen_rand, distancia, tempo_execucao))

        # Criar dataframe com os resultados
        df_customizado = pd.DataFrame(resultados_customizados, columns=["cxpb", "mutpb", "ngen", "Distância (km)", "Tempo (s)"])
        st.write("Resultados das 10 rodadas com configuração customizada:")
        st.write(df_customizado)

        # Gerar gráfico de dispersão
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df_customizado["Tempo (s)"], df_customizado["Distância (km)"], c='red', alpha=0.7)
        ax.set_title("Relação entre Tempo e Distância")
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Distância (km)")
        st.pyplot(fig)

        # Criar um arquivo Excel com duas abas
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_padrao.to_excel(writer, sheet_name="Resultados Padrão", index=False)
            df_customizado.to_excel(writer, sheet_name="Resultados Customizados", index=False)

        # Disponibilizar o arquivo Excel para download
        st.download_button(
            label="📥 Baixar Resultados Completos (Excel)",
            data=output.getvalue(),
            file_name="resultados_analise.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"
        )
