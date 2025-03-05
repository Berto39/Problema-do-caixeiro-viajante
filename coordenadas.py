import pandas as pd
import random

# Definição dos limites geográficos aproximados de Goiânia
limites = {
    "lat_min": -16.8000,
    "lat_max": -16.6000,
    "lon_min": -49.4000,
    "lon_max": -49.1000
}

# Geração de 20 endereços fictícios
enderecos = []
for i in range(1, 21):
    nome = f"Entrega {i}"
    lat = random.uniform(limites["lat_min"], limites["lat_max"])
    lon = random.uniform(limites["lon_min"], limites["lon_max"])
    enderecos.append([nome, lat, lon])

# Criar DataFrame e salvar como CSV
df = pd.DataFrame(enderecos, columns=["Nome", "Latitude", "Longitude"])
df.to_csv("enderecos_goiania.csv", index=False)

print("Arquivo CSV gerado com sucesso!")
