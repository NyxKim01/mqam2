"""Converte os resultados dos blocos em tabelas LaTeX (formato numérico brasileiro)."""
import math
from pathlib import Path

import numpy as np

TABELAS = Path("tabelas")
TABELAS.mkdir(exist_ok=True)

NOMES_COEF = {
    "Intercept": "Intercepto",
    "cidade[T.Rio de Janeiro]": "Cidade: Rio de Janeiro",
    "cidade[T.Belo Horizonte]": "Cidade: Belo Horizonte",
    "cidade[T.Porto Alegre]": "Cidade: Porto Alegre",
    "cidade[T.Campinas]": "Cidade: Campinas",
    "tipo[T.Casa]": "Tipo: Casa",
    "mobilia[T.Mobiliado]": "Mobília: Mobiliado",
    "animal[T.Não aceita]": "Animais: Não aceita",
    "log_area": r"$\log_{10}(\text{Área})$",
    "banheiros": "Banheiros",
    "vagas": "Vagas",
    "andar": "Andar",
    "bc_cond": "BC(Condomínio)",
    "bc_iptu": "BC(IPTU)",
}
NOMES_TERMOS = {
    "cidade": "Cidade", "tipo": "Tipo", "mobilia": "Mobília", "animal": "Animais",
    "log_area": r"$\log_{10}(\text{Área})$", "quartos": "Quartos", "banheiros": "Banheiros",
    "vagas": "Vagas", "andar": "Andar", "bc_cond": "BC(Condomínio)", "bc_iptu": "BC(IPTU)",
}
UNIDADES_EFEITO = {
    "log_area": "+10\\% de área", "banheiros": "+1 banheiro", "vagas": "+1 vaga", "andar": "+1 andar",
    "bc_cond": "1.º $\\to$ 3.º quartil", "bc_iptu": "1.º $\\to$ 3.º quartil",
}


def num(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return r"\NA"
    texto = f"{x:,.{d}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def cient(x, d=3):
    if x == 0:
        return r"$<10^{-300}$"
    expoente = int(math.floor(math.log10(abs(x))))
    mantissa = x / 10 ** expoente
    return rf"${num(mantissa, d).replace(',', '{,}')}\times10^{{{expoente}}}$"


def pv(p):
    return r"$<0,0001$" if p < 0.0001 else num(p, 4)


def salvar(nome, colunas, linhas, rodape=None):
    corpo = [r"\begin{tabular}{" + colunas + "}", r"\toprule"]
    for linha in linhas:
        bruta = linha.startswith((r"\midrule", r"\addlinespace"))
        corpo.append(linha if bruta else linha + r" \\")
    if rodape:
        corpo += [r"\midrule", rodape]
    corpo += [r"\bottomrule", r"\end{tabular}"]
    (TABELAS / f"{nome}.tex").write_text("\n".join(corpo) + "\n", encoding="utf-8")


# Registros implausíveis excluídos
def criterio(r):
    motivos = []
    if r["area"] > 10_000:
        motivos.append("área")
    if r["andar"] > 100:
        motivos.append("andar")
    if r["condominio"] > 20_000:
        motivos.append("condomínio")
    if r["iptu"] > 10_000:
        motivos.append("IPTU")
    return ", ".join(motivos)


linhas = ["Cidade & Área (m$^2$) & Quartos & Andar & Condomínio & Aluguel & IPTU & Critério", r"\midrule"]
for _, r in excluidos.sort_values(["cidade", "aluguel"]).iterrows():
    linhas.append(" & ".join([r["cidade"].replace("Sao Paulo", "São Paulo"), num(r["area"], 0), num(r["quartos"], 0),
                              num(r["andar"], 0), num(r["condominio"], 0), num(r["aluguel"], 0), num(r["iptu"], 0),
                              criterio(r)]))
salvar("excluidos", "lrrrrrrl", linhas)

# Box-Cox
ADOTADA = {"aluguel": r"$\log_{10}$", "area": r"$\log_{10}$", "quartos": "nenhuma", "banheiros": "nenhuma",
           "vagas": "nenhuma", "andar": "nenhuma", "condominio": r"BC($X+1$)", "iptu": r"BC($X+1$)"}
NOMES_VAR = {"aluguel": "Aluguel", "area": "Área", "quartos": "Quartos", "banheiros": "Banheiros",
             "vagas": "Vagas", "andar": "Andar", "condominio": "Condomínio", "iptu": "IPTU"}
COLUNA_FINAL = {"aluguel": "log_aluguel", "area": "log_area", "condominio": "bc_cond", "iptu": "bc_iptu"}
linhas = [r"Variável & Zeros & Assim. & $\lambda^*$ & Assim. BC & Assim. $\log_{10}$ & Adotada & Assim. final",
          r"\midrule"]
for v, r in tabela_boxcox.iterrows():
    final_col = COLUNA_FINAL.get(v, v)
    linhas.append(" & ".join([NOMES_VAR[v], num(r["zeros"], 0), num(r["assimetria"], 3), num(r["lambda_bc"], 3),
                              num(r["assim_bc"], 3), num(r["assim_log10"], 3), ADOTADA[v],
                              num(stats.skew(df[final_col]), 3)]))
salvar("boxcox", "lrrrrrlr", linhas)

# Trajetória do stepwise
linhas = [r"Passo & Ação & Termo & gl & $F$ parcial & $p$-valor & \Rtwoadj{} após", r"\midrule"]
for passo, r in trajetoria.iterrows():
    linhas.append(" & ".join([str(passo), r["acao"], NOMES_TERMOS[r["termo"]], str(r["gl"]), num(r["F"], 2),
                              cient(r["p"]) if r["p"] < 0.0001 else num(r["p"], 4), num(r["r2_aj"], 4)]))
salvar("stepwise", "clllrrr", linhas)

# Comparação completo x selecionado e importância
f_q, p_q = anova_lm(final, completo).iloc[1][["F", "Pr(>F)"]]
linhas = [r"Modelo & Termos & Parâmetros & \Rtwo{} & \Rtwoadj{} & AIC & BIC", r"\midrule"]
for nome_m, r in comparacao.iterrows():
    linhas.append(" & ".join([nome_m, num(r["termos"], 0), num(r["p"], 0), num(r["R2"], 4), num(r["R2_aj"], 4),
                              num(r["AIC"], 2), num(r["BIC"], 2)]))
salvar("comparacao_modelos", "lrrrrrr", linhas,
       rf"\multicolumn{{7}}{{l}}{{\footnotesize Teste $F$ parcial para incluir Quartos: $F={num(f_q, 4)}$; $p={num(p_q, 4)}$.}}\\")

linhas = [r"Termo & gl & $\Delta R^2$ ao remover & \% do \Rtwo{}", r"\midrule"]
for termo, delta in importancia.items():
    gl = int(round(final.df_model - ajustar([u for u in selecionados if u != termo]).df_model))
    linhas.append(" & ".join([NOMES_TERMOS[termo], str(gl), num(delta, 4), num(100 * delta / final.rsquared, 1)]))
salvar("importancia", "lrrr", linhas)

# Coeficientes e efeitos
linhas = [r"Termo & Estimativa & EP & $t$ & $p$-valor & IC 95\%", r"\midrule"]
for nome_c, r in coeficientes.iterrows():
    linhas.append(" & ".join([NOMES_COEF[nome_c], num(r["coef"], 5), num(r["ep"], 5), num(r["t"], 3), pv(r["p"]),
                              f"[{num(r['ic_inf'], 5)}; {num(r['ic_sup'], 5)}]"]))
salvar("coeficientes", "lrrrrl", linhas)

linhas = [r"Termo & Comparação & Variação (\%) & IC 95\% (\%)", r"\midrule"]
for nome_c, r in efeitos.iterrows():
    base = nome_c.split("[")[0]
    comparacao_txt = "vs. referência" if "[" in nome_c else UNIDADES_EFEITO[base]
    linhas.append(" & ".join([NOMES_COEF[nome_c], comparacao_txt, num(r["efeito"], 1),
                              f"[{num(r['ic_inf'], 1)}; {num(r['ic_sup'], 1)}]"]))
salvar("efeitos", "llrl", linhas)

# ANOVA da regressão
sq_reg, sq_res = modelo.ess, modelo.ssr
gl_reg, gl_res = int(modelo.df_model), int(modelo.df_resid)
linhas = [r"Fonte & gl & SQ & QM & $F$ & $p$-valor", r"\midrule",
          " & ".join(["Regressão", str(gl_reg), num(sq_reg, 3), num(sq_reg / gl_reg, 4), num(modelo.fvalue, 2),
                      pv(modelo.f_pvalue)]),
          " & ".join(["Resíduos", num(gl_res, 0), num(sq_res, 3), num(sq_res / gl_res, 5), r"\NA", r"\NA"]),
          " & ".join(["Total", num(gl_reg + gl_res, 0), num(modelo.centered_tss, 3), r"\NA", r"\NA", r"\NA"])]
salvar("anova", "lrrrrr", linhas,
       rf"\multicolumn{{6}}{{l}}{{\footnotesize $S={num(np.sqrt(modelo.scale), 4)}$; \Rtwo$={num(modelo.rsquared, 4)}$; "
       rf"\Rtwoadj$={num(modelo.rsquared_adj, 4)}$.}}\\")

# Resumo OLS (mesmo formato da Atividade 1)
from statsmodels.stats.stattools import jarque_bera, omni_normtest

omni, p_omni = omni_normtest(modelo.resid)
jb, p_jb, assim, curt = jarque_bera(modelo.resid)
linhas = [r"Estatística & Valor & Estatística & Valor", r"\midrule",
          rf"Observações & {num(modelo.nobs, 0)} & gl modelo/resíduos & {gl_reg}/{num(gl_res, 0)}",
          rf"\Rtwo{{}} & {num(modelo.rsquared, 4)} & \Rtwoadj{{}} & {num(modelo.rsquared_adj, 4)}",
          rf"$F$ & {num(modelo.fvalue, 2)} & $p(F)$ & {cient(modelo.f_pvalue)}",
          rf"Log-verossimilhança & {num(modelo.llf, 3)} & AIC & {num(modelo.aic, 3)}",
          rf"BIC & {num(modelo.bic, 3)} & Número de condição & {num(modelo.condition_number, 1)}",
          rf"Omnibus & {num(omni, 3)} & $p(\text{{Omnibus}})$ & {cient(p_omni)}",
          rf"JB & {num(jb, 3)} & $p(\text{{JB}})$ & {cient(p_jb)}",
          rf"Assimetria & {num(assim, 3)} & Curtose & {num(curt, 3)}",
          rf"DW (ordem do arquivo) & {num(durbin_watson(modelo.resid), 3)} & Covariância & Clássica"]
salvar("resumo_ols", "lrlr", linhas)

# Testes sobre os resíduos
linhas = [r"Premissa & Teste & Estatística & $p$-valor", r"\midrule"]
for i, r in testes.iterrows():
    if i > 0 and r["grupo"] != testes.loc[i - 1, "grupo"]:
        linhas.append(r"\addlinespace")
    # O p-valor de Lilliefors vem de uma tabela e é truncado em 0,001
    p_txt = r"$\le 0,001$" if r["teste"].startswith("Kolmogorov") and r["p"] <= 0.001 else cient(r["p"])
    linhas.append(" & ".join([r["grupo"], r["teste"].replace("-", "--"), num(r["estatistica"], 4), p_txt]))
salvar("testes", "llrr", linhas)

# Resíduos por decil dos valores preditos
linhas = [r"Decil & Predito médio & $10^{\text{predito}}$ (R\$) & Média res. & DP res. & \% com aluguel $\geq$ R\$ 15 mil",
          r"\midrule"]
for decil, r in por_decil.iterrows():
    linhas.append(" & ".join([str(decil), num(r["predito_medio"], 3), num(10 ** r["predito_medio"], 0),
                              num(r["media_residuo"], 4), num(r["dp_residuo"], 4), num(100 * r["prop_teto"], 1)]))
salvar("decis", "crrrrr", linhas)

# VIF e erros padrão robustos
linhas = [r"Termo & VIF & EP clássico & EP HC3 & Razão HC3/clássico & $p$ (HC3)", r"\midrule"]
for nome_c, r in comparacao_ep.iterrows():
    linhas.append(" & ".join([NOMES_COEF[nome_c], num(vif.get(nome_c, np.nan), 2) if nome_c in vif else r"\NA",
                              num(r["ep_classico"], 5), num(r["ep_hc3"], 5), num(r["razao"], 3), pv(r["p_hc3"])]))
salvar("vif_robusto", "lrrrrr", linhas)

# Registros mais influentes
linhas = [r"Cidade & Tipo & Área & Banh. & Andar & Condomínio & Aluguel & IPTU & $D$ & $h$ & $r^*$", r"\midrule"]
for _, r in diag.nlargest(5, "cook").iterrows():
    linhas.append(" & ".join([r["cidade"], r["tipo"].replace("Apartamento", "Apto."), num(r["area"], 0), num(r["banheiros"], 0), num(r["andar"], 0),
                              num(r["condominio"], 0), num(r["aluguel"], 0), num(r["iptu"], 0), num(r["cook"], 4),
                              num(r["alav"], 4), num(r["rstud"], 2)]))
salvar("influentes", "llrrrrrrrrr", linhas)

# Validação cruzada
ROTULOS_VAL = {"R2_log": r"\Rtwo{} fora da amostra, escala $\log_{10}$", "RMSE_log": r"RMSE, escala $\log_{10}$",
               "R2_reais": r"\Rtwo{} fora da amostra, em R\$", "RMSE_reais": r"RMSE (R\$)",
               "EAM_reais": r"Erro absoluto médio (R\$)", "EPAM_mediano": r"Erro percentual absoluto mediano (\%)",
               "negativas": "Previsões não positivas"}
CASAS = {"R2_log": 4, "RMSE_log": 4, "R2_reais": 4, "RMSE_reais": 0, "EAM_reais": 0, "EPAM_mediano": 1, "negativas": 0}
linhas = [r"Métrica (validação cruzada, 10 partes) & Modelo final & Sem transformações", r"\midrule"]
for metrica, r in validacao.iterrows():
    linhas.append(" & ".join([ROTULOS_VAL[metrica], num(r["Final (log10)"], CASAS[metrica]),
                              num(r["Sem transformações"], CASAS[metrica])]))
salvar("validacao", "lrr", linhas)
print("Tabelas gravadas em", TABELAS.resolve())
