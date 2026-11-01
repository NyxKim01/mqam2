NOMES = {
    "city": "cidade", "area": "area", "rooms": "quartos", "bathroom": "banheiros",
    "parking_spaces": "vagas", "floor": "andar", "animal": "animal", "furniture": "mobilia",
    "hoa_(BRL)": "condominio", "rent_amount_(BRL)": "aluguel", "property_tax_(BRL)": "iptu",
    "fire_insurance_(BRL)": "seguro", "total_(BRL)": "total",
}
df = bruto.rename(columns=NOMES)
df["andar"] = df["andar"].astype(int)
n_original = len(df)

# Registros idênticos em todas as 13 colunas (anúncio coletado mais de uma vez)
duplicado = df.duplicated()
df = df[~duplicado]

# Valores fisicamente ou economicamente implausíveis (erros de digitação ou de coleta)
implausivel = (
    (df["area"] > 10_000) | (df["andar"] > 100)
    | (df["condominio"] > 20_000) | (df["iptu"] > 10_000)
)
excluidos = df.loc[implausivel, ["cidade", "area", "quartos", "andar", "condominio", "aluguel", "iptu"]]
print(excluidos)
df = df[~implausivel].reset_index(drop=True)

print(f"Original: {n_original} | duplicados: {duplicado.sum()} | "
      f"implausíveis: {implausivel.sum()} | final: {len(df)}")
