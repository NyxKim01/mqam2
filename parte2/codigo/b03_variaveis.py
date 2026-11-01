# Seguro e total são calculados a partir do próprio aluguel: vazamento da resposta
print(df[["aluguel", "seguro", "total"]].corr().round(3))
print((df["seguro"] / df["aluguel"]).describe().round(4))
df = df.drop(columns=["seguro", "total"])

# Categorias com nomes em português e categorias de referência na primeira posição
df["cidade"] = pd.Categorical(
    df["cidade"].replace({"Sao Paulo": "São Paulo"}),
    categories=["São Paulo", "Rio de Janeiro", "Belo Horizonte", "Porto Alegre", "Campinas"])
df["animal"] = pd.Categorical(df["animal"].map({"acept": "Aceita", "not acept": "Não aceita"}),
                              categories=["Aceita", "Não aceita"])
df["mobilia"] = pd.Categorical(df["mobilia"].map({"not furnished": "Não mobiliado", "furnished": "Mobiliado"}),
                               categories=["Não mobiliado", "Mobiliado"])

# Andar "-" (codificado como 0 no OpenML) indica casa; nos apartamentos, andar >= 1
df["tipo"] = pd.Categorical(np.where(df["andar"] == 0, "Casa", "Apartamento"),
                            categories=["Apartamento", "Casa"])

for coluna in ["cidade", "tipo", "animal", "mobilia"]:
    print(df[coluna].value_counts(), "\n")
print(pd.crosstab(df["tipo"], df["condominio"] == 0, margins=True))
print(pd.crosstab(df["tipo"], df["iptu"] == 0, margins=True))
