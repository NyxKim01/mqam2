modelo = final
print(modelo.summary())

ic = modelo.conf_int()
coeficientes = pd.DataFrame({"coef": modelo.params, "ep": modelo.bse, "t": modelo.tvalues,
                             "p": modelo.pvalues, "ic_inf": ic[0], "ic_sup": ic[1]})
print(coeficientes.round(5))
print(f"R2 = {modelo.rsquared:.4f} | R2 ajustado = {modelo.rsquared_adj:.4f} | "
      f"F({modelo.df_model:.0f}, {modelo.df_resid:.0f}) = {modelo.fvalue:.2f} | p = {modelo.f_pvalue:.3g}")


def variacao(b):
    """Variação percentual na mediana condicional do aluguel para um acréscimo b em log10."""
    return 100 * (10 ** np.asarray(b) - 1)


# Efeitos na escala original: categorias e contagens (+1 unidade), área (+10%),
# condomínio e IPTU (do 1.o ao 3.o quartil da variável original)
efeitos = {}
for nome in modelo.params.index.drop("Intercept"):
    escala = 1.0
    if nome == "log_area":
        escala = np.log10(1.10)
    elif nome in ("bc_cond", "bc_iptu"):
        original, lam = ("condominio", LAMBDA_COND) if nome == "bc_cond" else ("iptu", LAMBDA_IPTU)
        q1, q3 = df[original].quantile([0.25, 0.75])
        escala = stats.boxcox(np.array([q3 + 1.0]), lam)[0] - stats.boxcox(np.array([q1 + 1.0]), lam)[0]
    efeitos[nome] = dict(escala=escala, efeito=variacao(modelo.params[nome] * escala),
                         ic_inf=variacao(ic.loc[nome, 0] * escala), ic_sup=variacao(ic.loc[nome, 1] * escala))
efeitos = pd.DataFrame(efeitos).T
print(efeitos.round(2))
print(df[["condominio", "iptu"]].quantile([0.25, 0.75]))
