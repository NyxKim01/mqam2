from statsmodels.stats.diagnostic import het_breuschpagan, het_goldfeldquandt, lilliefors, normal_ad
from statsmodels.stats.stattools import durbin_watson

r = modelo.resid.to_numpy()
X = modelo.model.exog

normalidade = {
    "Shapiro-Wilk": stats.shapiro(r)[:2],
    "Kolmogorov-Smirnov (Lilliefors)": lilliefors(r, dist="norm"),
    "Anderson-Darling": normal_ad(r),
    "Jarque-Bera": tuple(stats.jarque_bera(r)),
    "D'Agostino-Pearson (Omnibus)": tuple(stats.normaltest(r)),
}

ordem = np.argsort(modelo.fittedvalues.to_numpy())  # Goldfeld-Quandt: ordenação pelos preditos
grupos_cidade = [r[(df["cidade"] == c).to_numpy()] for c in df["cidade"].cat.categories]
variancia = {
    "Breusch-Pagan (Koenker)": het_breuschpagan(r, X)[:2],
    "Goldfeld-Quandt (preditos)": het_goldfeldquandt(modelo.model.endog[ordem], X[ordem], drop=0.2,
                                                     result_object=False)[:2],
    "Brown-Forsythe (cidades)": tuple(stats.levene(*grupos_cidade, center="median")),
}

testes = pd.DataFrame(
    [dict(grupo="Normalidade", teste=k, estatistica=v[0], p=v[1]) for k, v in normalidade.items()]
    + [dict(grupo="Variância", teste=k, estatistica=v[0], p=v[1]) for k, v in variancia.items()])
print(testes)
print(f"Durbin-Watson (ordem do arquivo) = {durbin_watson(r):.3f}")
