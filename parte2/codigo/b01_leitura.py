import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import arff
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

import graficos as gr

# Brazilian_houses (OpenML, ID 42688) = houses_to_rent_v2.csv (Kaggle)
dados, meta = arff.loadarff("dados/Brazilian_houses.arff")
bruto = pd.DataFrame(dados)
for coluna in bruto.select_dtypes(include="object"):
    bruto[coluna] = bruto[coluna].str.decode("utf-8")

print(bruto.shape)
print(bruto.dtypes)
print(bruto.head())
