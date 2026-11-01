gr.diagnostico(modelo, "diagnostico_final")
gr.residuos_preditores(
    modelo, df,
    {t: ROTULOS[t] for t in ["log_area", "banheiros", "vagas", "andar", "bc_cond", "bc_iptu"] if t in selecionados},
    {t: CATEGORICAS[t] for t in ["cidade", "tipo", "mobilia"] if t in selecionados},
    "residuos_preditores")

residuos = modelo.resid
print(residuos.describe().round(4))
print(f"Assimetria = {stats.skew(residuos):.3f} | curtose = {stats.kurtosis(residuos, fisher=False):.3f}")
print(f"Fração com |resíduo| > 2S: {np.mean(np.abs(residuos) > 2 * np.sqrt(modelo.scale)):.4f}")
print(df.assign(residuo=residuos).groupby("cidade", observed=True)["residuo"].agg(["count", "mean", "std"]).round(4))

# Concentração de anúncios no valor máximo usual (possível teto da coleta)
print(f"Aluguel = R$ 15.000: {(df['aluguel'] == 15_000).sum()} | acima de R$ 15.000: {(df['aluguel'] > 15_000).sum()}")
decis = df.assign(residuo=residuos, predito=modelo.fittedvalues, teto=df["aluguel"] >= 15_000,
                  decil=pd.qcut(modelo.fittedvalues, 10, labels=range(1, 11)))
por_decil = decis.groupby("decil", observed=True).agg(
    predito_medio=("predito", "mean"), media_residuo=("residuo", "mean"),
    dp_residuo=("residuo", "std"), prop_teto=("teto", "mean"))
print(por_decil.round(4))
