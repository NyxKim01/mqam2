CANDIDATOS = ["cidade", "tipo", "mobilia", "animal", "log_area", "quartos", "banheiros",
              "vagas", "andar", "bc_cond", "bc_iptu"]
ALFA_E = ALFA_R = 0.15  # STAT501, Lição 10.2


def ajustar(termos: list[str]):
    lado_direito = " + ".join(t for t in CANDIDATOS if t in termos) or "1"
    return smf.ols(f"log_aluguel ~ {lado_direito}", data=df).fit()


def teste_f_parcial(reduzido, completo):
    """Teste F parcial de um termo (uma ou mais colunas): retorna F, p, log(p) e gl."""
    gl = int(round(completo.df_model - reduzido.df_model))
    f = ((reduzido.ssr - completo.ssr) / gl) / (completo.ssr / completo.df_resid)
    return f, stats.f.sf(f, gl, completo.df_resid), stats.f.logsf(f, gl, completo.df_resid), gl


selecionados, trajetoria = [], []
for _ in range(50):
    atual = ajustar(selecionados)
    entradas = [(*teste_f_parcial(atual, ajustar(selecionados + [t])), t)
                for t in CANDIDATOS if t not in selecionados]
    if not entradas:
        break
    f, p, logp, gl, termo = min(entradas, key=lambda e: e[2])  # menor p (em escala log)
    if p >= ALFA_E:
        break
    selecionados.append(termo)
    trajetoria.append(dict(acao="Entrada", termo=termo, gl=gl, F=f, p=p,
                           r2_aj=ajustar(selecionados).rsquared_adj))
    while len(selecionados) > 1:
        atual = ajustar(selecionados)
        saidas = [(*teste_f_parcial(ajustar([u for u in selecionados if u != s]), atual), s)
                  for s in selecionados]
        f, p, logp, gl, termo = max(saidas, key=lambda e: e[2])  # maior p
        if p <= ALFA_R:
            break
        selecionados.remove(termo)
        trajetoria.append(dict(acao="Saída", termo=termo, gl=gl, F=f, p=p,
                               r2_aj=ajustar(selecionados).rsquared_adj))

trajetoria = pd.DataFrame(trajetoria, index=pd.RangeIndex(1, len(trajetoria) + 1, name="passo"))
print(trajetoria)
nao_selecionados = [t for t in CANDIDATOS if t not in selecionados]
print("Selecionados:", selecionados, "| não selecionados:", nao_selecionados)

completo, final = ajustar(CANDIDATOS), ajustar(selecionados)
print(anova_lm(final, completo))
comparacao = pd.DataFrame({m: dict(termos=len(t), p=int(r.df_model) + 1, R2=r.rsquared, R2_aj=r.rsquared_adj,
                                   AIC=r.aic, BIC=r.bic)
                           for m, t, r in [("Completo", CANDIDATOS, completo),
                                           ("Selecionado", selecionados, final)]}).T
print(comparacao)

# Contribuição de cada termo: queda no R2 ao retirá-lo do modelo selecionado
importancia = pd.Series({t: final.rsquared - ajustar([u for u in selecionados if u != t]).rsquared
                         for t in selecionados}).sort_values(ascending=False)
print(importancia.round(4))
