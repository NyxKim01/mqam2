NUMERICAS = ["aluguel", "area", "quartos", "banheiros", "vagas", "andar", "condominio", "iptu"]

linhas = []
for v in NUMERICAS:
    x = df[v].astype(float)
    desloc = 1.0 if (x == 0).any() else 0.0  # Box-Cox exige valores positivos
    _, lam = stats.boxcox(x + desloc)
    linhas.append(dict(variavel=v, zeros=int((x == 0).sum()), deslocamento=desloc,
                       assimetria=stats.skew(x), lambda_bc=lam,
                       assim_bc=stats.skew(stats.boxcox(x + desloc, lam)),
                       assim_log10=stats.skew(np.log10(x + desloc)) if (x + desloc).min() > 0 else np.nan))
tabela_boxcox = pd.DataFrame(linhas).set_index("variavel")
print(tabela_boxcox.round(3))

# Transformações adotadas
LAMBDA_COND = tabela_boxcox.loc["condominio", "lambda_bc"]
LAMBDA_IPTU = tabela_boxcox.loc["iptu", "lambda_bc"]
df["log_aluguel"] = np.log10(df["aluguel"])
df["log_area"] = np.log10(df["area"])
df["bc_cond"] = stats.boxcox(df["condominio"] + 1.0, LAMBDA_COND)
df["bc_iptu"] = stats.boxcox(df["iptu"] + 1.0, LAMBDA_IPTU)

TRANSFORMADAS = ["log_aluguel", "log_area", "quartos", "banheiros", "vagas", "andar", "bc_cond", "bc_iptu"]
print(df[TRANSFORMADAS].agg(["mean", "std", "min", "max", "skew"]).T.round(3))
