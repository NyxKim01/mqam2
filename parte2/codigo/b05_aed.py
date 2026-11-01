ROTULOS = {
    "log_aluguel": "log10(Aluguel)", "log_area": "log10(Área)", "quartos": "Quartos",
    "banheiros": "Banheiros", "vagas": "Vagas", "andar": "Andar",
    "bc_cond": "Box-Cox(Condomínio)", "bc_iptu": "Box-Cox(IPTU)",
}
CATEGORICAS = {"cidade": "Cidade", "tipo": "Tipo de imóvel", "animal": "Animais", "mobilia": "Mobília"}

gr.histogramas(df, {"aluguel": "Aluguel (R$)", "area": "Área (m²)", "condominio": "Condomínio (R$)",
                    "iptu": "IPTU (R$)"}, "hist_originais", colunas_grade=2)
gr.histogramas(df, ROTULOS, "hist_transformadas")
gr.barras(df, CATEGORICAS, "barras_categoricas")
gr.matriz_dispersao(df, ROTULOS, "matriz_dispersao")
gr.boxplots(df, "log_aluguel", "log10(Aluguel)", CATEGORICAS, "boxplots_categoricas")

print(df[TRANSFORMADAS].corr().round(3))
for coluna in CATEGORICAS:
    print(df.groupby(coluna, observed=True)["aluguel"].agg(["count", "median", "mean"]).round(0), "\n")
