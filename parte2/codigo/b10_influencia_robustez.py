from statsmodels.stats.outliers_influence import variance_inflation_factor

# Multicolinearidade
Xd = pd.DataFrame(modelo.model.exog, columns=modelo.model.exog_names)
vif = pd.Series({c: variance_inflation_factor(Xd.values, i)
                 for i, c in enumerate(Xd.columns) if c != "Intercept"})
print(vif.round(2))
print(f"Número de condição = {modelo.condition_number:.1f}")

# Observações influentes
infl = modelo.get_influence()
n, p = int(modelo.nobs), int(modelo.df_model) + 1
diag = df.assign(cook=infl.cooks_distance[0], alav=infl.hat_matrix_diag,
                 rstud=infl.resid_studentized_external)
print(f"max D = {diag['cook'].max():.4f} | D > 4/n: {(diag['cook'] > 4 / n).sum()} | "
      f"h > 2p/n: {(diag['alav'] > 2 * p / n).sum()} | |rstud| > 3: {(diag['rstud'].abs() > 3).sum()}")
print(diag.nlargest(5, "cook")[["cidade", "tipo", "area", "quartos", "banheiros", "andar",
                                "condominio", "aluguel", "iptu", "cook", "alav", "rstud"]])
gr.influencia(modelo, "influencia")

# Sensibilidade: reajuste sem os registros com D > 4/n (somente para comparação)
sem_influentes = smf.ols(modelo.model.formula, data=df[diag["cook"] <= 4 / n]).fit()
desvio = ((sem_influentes.params - modelo.params) / modelo.bse).abs()
print(f"R2 sem influentes = {sem_influentes.rsquared:.4f} | maior mudança = {desvio.max():.2f} EP ({desvio.idxmax()})")
print(pd.DataFrame({"todos": modelo.params, "sem_influentes": sem_influentes.params}).round(4))

# Erros padrão robustos à heterocedasticidade (HC3)
robusto = smf.ols(modelo.model.formula, data=df).fit(cov_type="HC3")
comparacao_ep = pd.DataFrame({"ep_classico": modelo.bse, "ep_hc3": robusto.bse,
                              "razao": robusto.bse / modelo.bse,
                              "p_classico": modelo.pvalues, "p_hc3": robusto.pvalues})
print(comparacao_ep.round(5))
