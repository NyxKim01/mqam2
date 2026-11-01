# Modelo de referência: mesmas preditoras candidatas, sem nenhuma transformação
formula_original = ("aluguel ~ cidade + tipo + mobilia + animal + area + quartos + banheiros"
                    " + vagas + andar + condominio + iptu")
original = smf.ols(formula_original, data=df).fit()
print(f"Escala original: R2 = {original.rsquared:.4f} | R2 ajustado = {original.rsquared_adj:.4f} | "
      f"assimetria = {stats.skew(original.resid):.3f} | "
      f"curtose = {stats.kurtosis(original.resid, fisher=False):.3f}")
gr.comparacao_escalas(original, modelo, "comparacao_escalas")

# Validação cruzada em 10 partes: previsões sempre feitas em registros fora do ajuste
dobras = np.random.default_rng(2026).permutation(len(df)) % 10
previsoes = pd.DataFrame(index=df.index, columns=["log_final", "rs_final", "rs_original"], dtype=float)
for k in range(10):
    treino, teste = df[dobras != k], df[dobras == k]
    ajuste = smf.ols(modelo.model.formula, data=treino).fit()
    smearing = np.mean(10 ** ajuste.resid)  # retransformação de Duan (1983)
    previsoes.loc[teste.index, "log_final"] = ajuste.predict(teste)
    previsoes.loc[teste.index, "rs_final"] = 10 ** ajuste.predict(teste) * smearing
    previsoes.loc[teste.index, "rs_original"] = smf.ols(formula_original, data=treino).fit().predict(teste)


def r2(y, y_hat):
    return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - np.mean(y)) ** 2)


y, y_log = df["aluguel"], df["log_aluguel"]
validacao = pd.DataFrame({
    "Final (log10)": dict(R2_log=r2(y_log, previsoes["log_final"]),
                          RMSE_log=np.sqrt(np.mean((y_log - previsoes["log_final"]) ** 2)),
                          R2_reais=r2(y, previsoes["rs_final"]),
                          RMSE_reais=np.sqrt(np.mean((y - previsoes["rs_final"]) ** 2)),
                          EAM_reais=np.mean(np.abs(y - previsoes["rs_final"])),
                          EPAM_mediano=100 * np.median(np.abs(y - previsoes["rs_final"]) / y),
                          negativas=int((previsoes["rs_final"] <= 0).sum())),
    "Sem transformações": dict(R2_log=np.nan, RMSE_log=np.nan,
                               R2_reais=r2(y, previsoes["rs_original"]),
                               RMSE_reais=np.sqrt(np.mean((y - previsoes["rs_original"]) ** 2)),
                               EAM_reais=np.mean(np.abs(y - previsoes["rs_original"])),
                               EPAM_mediano=100 * np.median(np.abs(y - previsoes["rs_original"]) / y),
                               negativas=int((previsoes["rs_original"] <= 0).sum())),
})
print(validacao.round(4))
print(f"Fator multiplicativo típico 10^S = {10 ** np.sqrt(modelo.scale):.3f}")
