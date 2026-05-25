# ======================================================================================================================================================

# Code for the article Manuel Verdú, Óscar Carchano & Sergio Belza González, ESG scores and Arbitrage Opportunities in Rights Issues: Evidence from International Equity Markets)
# Financial Internet Quarterly. 22:3. Pending of Publication.

# ======================================================================================================================================================

# To execute this code, you must have the file 'Verdu_Carchano_Belza_2026_Data.csv' available in ‘https://doi.org/10.5281/zenodo.18450859’.

# Once the data is available in the same directory, you only need to execute the code to obtain the results shown in the article.

# The article can be found at: Pending of Publication.

# ======================================================================================================================================================

#################### LIBRARIES TO USE ####################


import os
import warnings
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from scipy.stats import chi2
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV

from scipy import stats

warnings.filterwarnings("ignore")

#################### START OF AUXILIAR PROGRAMS ####################

def grangers_causality_matrix(data, variables, max_lag, test='ssr_chi2test', verbose=False):    
    """
    Crea una matriz con los resultados del Test de Causalidad de Granger.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            if r != c:  # Evitar pruebas de autocausalidad
                test_result = grangercausalitytests(data[[r, c]], max_lag, verbose=False)
                p_values = [round(test_result[i+1][0][test][1], 4) for i in range(max_lag)]
                min_p_value = np.min(p_values)
                df.loc[r, c] = min_p_value
    df.columns = [var + '_causes' for var in variables]
    df.index = [var + '_caused_by' for var in variables]
    return df

def granger_causality_indep_to_target(data, target, indep_vars, max_lag):
    """
    Aplica el Test de Causalidad de Granger a múltiples variables independientes sobre una variable objetivo.
    """
    results = {}
    for var in indep_vars:
        test_result = grangercausalitytests(data[[target, var]], max_lag, verbose=False)
        p_values = [round(test_result[i+1][0]['ssr_chi2test'][1], 4) for i in range(max_lag)]
        results[var] = min(p_values)  # Nos quedamos con el valor p mínimo
    return results

def bootstrap_elastic_net(X, Y, n_bootstrap=1000, alpha=0.05, l1_ratios=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99], 
                          alphas=np.logspace(-4, 1, 50), random_state=42):
    """
    Bootstrap para obtener intervalos de confianza de coeficientes Elastic Net.
    
    Parameters:
    -----------
    X : array-like, features
    Y : array-like, target
    n_bootstrap : int, número de iteraciones bootstrap
    alpha : float, nivel de significancia (0.05 para IC 95%)
    
    Returns:
    --------
    coef_mean : coeficientes promedio
    coef_ci_lower : límite inferior IC
    coef_ci_upper : límite superior IC
    coef_std : desviación estándar de coeficientes
    pct_nonzero : porcentaje de veces que el coeficiente fue != 0
    """
    np.random.seed(random_state)
    n_samples, n_features = X.shape
    boot_coefs = np.zeros((n_bootstrap, n_features))
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]
        Y_boot = Y[indices]
        
        # Estandardization
        scaler = StandardScaler()
        X_boot_scaled = scaler.fit_transform(X_boot)
        
        # Model adjustment
        model = ElasticNetCV(l1_ratio=l1_ratios, alphas=alphas, cv=5, max_iter=10000, n_jobs=-1)
        model.fit(X_boot_scaled, Y_boot)
        boot_coefs[i, :] = model.coef_

    print(f"Optimal alpha: {model.alpha_}")
    print(f"Optimal l1_ratio: {model.l1_ratio_}")
    
    # Calculus of statistics
    coef_mean = np.mean(boot_coefs, axis=0)
    coef_std = np.std(boot_coefs, axis=0)
    coef_ci_lower = np.percentile(boot_coefs, (alpha/2) * 100, axis=0)
    coef_ci_upper = np.percentile(boot_coefs, (1 - alpha/2) * 100, axis=0)
    pct_nonzero = np.mean(boot_coefs != 0, axis=0) * 100
    
    return coef_mean, coef_ci_lower, coef_ci_upper, coef_std, pct_nonzero

def print_bootstrap_results(var_names, coef_mean, coef_ci_lower, coef_ci_upper, coef_std, pct_nonzero):
    """Imprime resultados del bootstrap en formato tabla."""
    print('')
    print(f'{"Variable":<10} {"Coef":>10} {"Std":>10} {"CI 95% Lower":>12} {"CI 95% Upper":>12} {"% Non-Zero":>10} {"Signif":>8}')
    print('-' * 75)
    for i, name in enumerate(var_names):
        # Significative if the IC does not include 0
        signif = '*' if (coef_ci_lower[i] > 0 or coef_ci_upper[i] < 0) else ''
        print(f'{name:<10} {coef_mean[i]:>10.4f} {coef_std[i]:>10.4f} {coef_ci_lower[i]:>12.4f} {coef_ci_upper[i]:>12.4f} {pct_nonzero[i]:>10.1f} {signif:>8}')
    print('')
    print('* = IC 95% no incluye cero (estadísticamente significativo)')

def net(DAT, Model, n_bootstrap = 1000):
    
    X1, X2, X3, X4 = DAT['ESG'], DAT['ENV'], DAT['SOC'], DAT['GOV']
    X5, X6, X7, X8, X9, X10 = DAT['RAT'], DAT['DLR'], DAT['DIL'], DAT['BTM'], DAT['ILI'], DAT['HIG']

    # Create interaction term RAT * DIL
    X_RAT_DIL = X5 * X7

    if Model == 1:
        Y = DAT['RET'].values

    elif Model == 2:
        Y = DAT['ARB'].values

    DTBSX1 = pd.DataFrame({'ESG': X1, 
        'RAT': X5, 'DLR': X6, 'DIL': X7, 'RAT_DIL': X_RAT_DIL, 'BTM': X8, 'ILI': X9, 'HIG': X10})
    
    DTBSX2 = pd.DataFrame({'ENV': X2, 'SOC': X3, 'GOV': X4,
        'RAT': X5, 'DLR': X6, 'DIL': X7, 'RAT_DIL': X_RAT_DIL, 'BTM': X8, 'ILI': X9, 'HIG': X10})

    var_names1 = ['ESG', 'RAT', 'DLR', 'DIL', 'RAT_DIL', 'BTM', 'ILI', 'HIG']
    var_names2 = ['ENV', 'SOC', 'GOV', 'RAT', 'DLR', 'DIL', 'RAT_DIL', 'BTM', 'ILI', 'HIG']

    # Define hyperparameters
    l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    alphas = np.logspace(-4, 1, 50)

    print('\n=============== ESG General ===============\n')
    
    # Bootstrap for Model I
    coef_mean1, ci_lower1, ci_upper1, coef_std1, pct_nz1 = bootstrap_elastic_net(
        DTBSX1.values, Y, n_bootstrap = n_bootstrap, l1_ratios = l1_ratios, alphas = alphas
    )
    print_bootstrap_results(var_names1, coef_mean1, ci_lower1, ci_upper1, coef_std1, pct_nz1)

    print('\n=============== ESG Pillars ===============\n')

    # Bootstrap for Model II
    coef_mean2, ci_lower2, ci_upper2, coef_std2, pct_nz2 = bootstrap_elastic_net(
        DTBSX2.values, Y, n_bootstrap = n_bootstrap, l1_ratios = l1_ratios, alphas = alphas
    )
    print_bootstrap_results(var_names2, coef_mean2, ci_lower2, ci_upper2, coef_std2, pct_nz2)

def prob_validation(DATA, res, model):

    print(res.summary())
    print()

    # Obtain fitted values
    y = res.predict()
    Y = (y >= 0.5).astype(int)
    L = len(Y)

    # Distribution of arbitrages in real and fitted sample
    ARB_R = sum(DATA['ARB'])
    ARB_F = sum(Y)

    print('##### Arbitrages #####')
    print(f'Real Sample: {ARB_R} / {len(DATA["ARB"])} ({round(100 * ARB_R / len(DATA["ARB"]), 2)}%) - Fitted Sample: ({ARB_F} / {len(Y)}) ({100 * round(ARB_F / len(Y), 2)}%)')
    print()

    # Success of the estimation in predicting arbitrage opportunities

    pos, neg, tot = 0, 0, 0
    arb, narb = 0, 0

    for l in range(0, L):

        if DATA['ARB'][l] == 1:
            arb += 1

            if Y[l] == 1:

                pos += 1
                tot += 1

        else:

            narb += 1

            if Y[l] == 0:

                neg += 1
                tot += 1

    print('##### Prediction #####')
    print(f'Arbitrage: {pos} / {arb} ({round(100 * pos / arb, 2)}%) - No Arbitrage: {neg} / {narb} ({round(100 * neg / narb, 2)}%) - Total: {tot} / {L} ({round(100 * tot / L, 2)}%))')
    print()

    # Statistic Test

    S = []

    for l in range(0, L):

        if round(y[l], 0) != 0:
            s = (( DATA['ARB'][l] - y[l]) ** 2) / (y[l] * (1 - y[l]))
            S.append(s)

    S_Stat = sum(S)
    k = 7 if model == 1 else 10

    PVal = 1 - chi2.cdf(S_Stat, df = L - k)

    if PVal <= 0.01:
        result = '***'
    elif PVal <= 0.05:
        result = '**'
    elif PVal <= 0.1:
        result = '*'
    else:
        result = '-'

    print('##### S-Statistics #####')
    print(f'S: {round(S_Stat, 4)} - P-Value: {round(PVal, 4)} ({result})')
    print()

    # Determination coefficient with residuals

    SR = sum((DATA['ARB'] - y) ** 2)
    ST = sum((DATA['ARB'] - DATA['ARB'].mean()) ** 2)

    RHO = 1 - (SR / ST)

    print('##### Coefficient of Determination based in residuals #####')
    print(f'Rho: {round(RHO * 100, 4)}%')

#################### END OF AUXILIAR PROGRAMS ####################

#################### START OF THE CODE ####################

DATA = pd.read_csv('Verdu_Carchano_Belza_2025_Data.csv', sep = ';')

# Detect outliers in RET using MAD (Median Absolute Deviation) methodology
median_ret = DATA['RET'].median()
mad = np.median(np.abs(DATA['RET'] - median_ret))
# Using modified Z-score with MAD (threshold typically 3.5)
modified_z_scores = 0.6745 * (DATA['RET'] - median_ret) / mad
DATA['OUT'] = (np.abs(modified_z_scores) > 2).astype(int)
DATA1 = DATA[DATA['OUT'] == 0]

CON = ['AFR', 'AME', 'ASI', 'EUR']

COU = ['KWT', 'MUS', 'MAR', 'OMN', 'SAU', 'ZAF', 'ARE'
, 'BRA', 'CAN', 'COL', 'MEX', 'USA'
, 'AUS', 'CHN', 'HKG', 'IND', 'IDN', 'JPN', 'MYS', 'NZL', 'PHL', 'KOR', 'TWN', 'THA', 'VNM'
, 'AUT', 'BEL', 'DNK', 'FIN', 'FRA', 'DEU', 'GRC', 'ITA', 'NLD', 'NOR', 'ESP', 'RUS', 'SWE', 'CHE', 'TUR', 'GBR']

SEC = ['ACA', 'BAS', 'CYC', 'NCY', 'ENE', 'FIN', 'HEA', 'IND', 'EST', 'TEC', 'UTI']

########## STATISTICAL DESCRIPTION OF THE SAMPLE ##########

print('========================= Observations per Country =========================')
# Count observations for each country
for country in COU:
    count = (DATA['COU'] == country).sum()
    out_c = ((DATA['COU'] == country) & (DATA['OUT'] == 1)).sum()
    print(f"{country}: {count} ({out_c})")

print()
print('========================= Observations per Sector =========================')
# Count observations for each sector
for sector in SEC:
    count = (DATA['SEC'] == sector).sum()
    out_s = ((DATA['SEC'] == sector) & (DATA['OUT'] == 1)).sum()
    print(f"{sector}: {count} ({out_s})")

print('\n========================= Main Statistics =========================')
# Calculate descriptive statistics for the specified series
series_to_analyze1 = ['ARB', 'ESG', 'ENV', 'SOC', 'GOV']

# Create a dictionary to store statistics
stats_dict = {}

for series in series_to_analyze1:
    stats_dict[series] = {
        'Mean': DATA[series].mean(),
        'Std Dev': DATA[series].std(),
        'Median': DATA[series].median(),
        'N Obs': DATA[series].count(),
        'Maximum': DATA[series].max(),
        'Minimum': DATA[series].min(),
        'Skewness': DATA[series].skew(),
        'Kurtosis': DATA[series].kurtosis()
    }

# Convert to DataFrame for better display
import pandas as pd
stats_df = pd.DataFrame(stats_dict).T
stats_df = stats_df.round(4)
print(stats_df)
print()
# Calculate descriptive statistics for the returns (without outliers)
series_to_analyze2 = ['RET']

# Create a dictionary to store statistics
stats_dict = {}

for series in series_to_analyze2:
    stats_dict[series] = {
        'Mean': DATA1[series].mean(),
        'Std Dev': DATA1[series].std(),
        'Median': DATA1[series].median(),
        'N Obs': DATA1[series].count(),
        'Maximum': DATA1[series].max(),
        'Minimum': DATA1[series].min(),
        'Skewness': DATA1[series].skew(),
        'Kurtosis': DATA1[series].kurtosis()
    }

# Convert to DataFrame for better display
import pandas as pd
stats_df = pd.DataFrame(stats_dict).T
stats_df = stats_df.round(4)
print(stats_df)

print('\n========================= Arbitrage Statistics =========================')
# Calculate descriptive statistics for the specified series
ESG_series = ['A', 'B', 'C', 'D']

# Create a dictionary to store statistics
stats_dict = {}

for series in ESG_series:
    stats_dict[series] = {
        'N': (DATA['GRA'] == series).sum(),
        'ARB': ((DATA['GRA'] == series) & (DATA['ARB'] == 1)).sum(),
		'nARB': ((DATA['GRA'] == series) & (DATA['ARB'] == 0)).sum(),
		'%_ARB': ((DATA['GRA'] == series) & (DATA['ARB'] == 1)).sum() / (((DATA['GRA'] == series) & (DATA['ARB'] == 1)).sum() + ((DATA['GRA'] == series) & (DATA['ARB'] == 0)).sum()),
		'%_nARB': ((DATA['GRA'] == series) & (DATA['ARB'] == 0)).sum() / (((DATA['GRA'] == series) & (DATA['ARB'] == 1)).sum() + ((DATA['GRA'] == series) & (DATA['ARB'] == 0)).sum()),
        'OUT': ((DATA['GRA'] == series) & (DATA['OUT'] == 1)).sum()
    }

# Convert to DataFrame for better display
import pandas as pd
stats_df = pd.DataFrame(stats_dict).T
stats_df = stats_df.round(4)
print(stats_df)

# Total Sample:

N = DATA['ARB'].count()
ARB = ((DATA['ARB'] == 1)).sum()
nARB = ((DATA['ARB'] == 0)).sum()
p_ARB = ARB / N
p_nARB = nARB / N
OUT = ((DATA['OUT'] == 1)).sum()

print(f'Total Sample ({N}) - Arbitrage: {ARB} ({p_ARB.round(4)}) - Non Arbitrage: {nARB} ({p_nARB.round(4)}) - Outliers: {OUT}')

########## ECONOMETRIC ANALYSIS ##########

##### ECONOMETRIC MODELS #####

# Definition of formulas for econometric models
FOR_O1 = 'RET ~ ESG + RAT + DLR + RAT * DIL + BTM + ILI + HIG'
FOR_O2 = 'RET ~ ENV + SOC + GOV + RAT + DLR + RAT * DIL + BTM + ILI + HIG'
FOR_P1 = 'ARB ~ ESG + RAT + DLR + RAT * DIL + BTM + ILI + HIG'
FOR_P2 = 'ARB ~ ENV + SOC + GOV + RAT + DLR + RAT * DIL + BTM + ILI + HIG'

print('')
print('##### OLS Model - Returns - General ESG #####')

mod = smf.ols(FOR_O1, DATA1)
res = mod.fit(cov_type = 'HC1') #white robustness.
print(res.summary())

print('')
print('##### Logit Model - Arbitrage - General ESG #####')

mod_l = smf.logit(FOR_P1, DATA)
res_l = mod_l.fit(cov_type = 'HC1') #white robustness.
prob_validation(DATA, res_l, 1)

print('')
print('##### Probit Model - Arbitrage - Geeral ESG #####')

mod_p = smf.probit(FOR_P1, DATA)
res_p = mod_p.fit(cov_type = 'HC1') #white robustness.
prob_validation(DATA, res_p, 1)


print('')
print('##### OLS Model - Returns - ESG Pilars #####')

mod = smf.ols(FOR_O2, DATA1)
res = mod.fit(cov_type = 'HC1') #white robustness.
print(res.summary())

print('')
print('##### Logit Model - Arbitrage - ESG Pilars #####')

mod_l = smf.logit(FOR_P2, DATA)
res_l = mod_l.fit(cov_type = 'HC1') #white robustness.
prob_validation(DATA, res_l, 2)

print('')
print('##### Probit Model - Arbitrage - ESG Pilars #####')

mod_p = smf.probit(FOR_P2, DATA)
res_p = mod_p.fit(cov_type = 'HC1') #white robustness.
prob_validation(DATA, res_p, 2)

##### GRANGER CAUSALITY TEST #####

### Parámetros y Datos ###

max_lag = 3

var_ret = ['RET', 'ENV', 'SOC', 'GOV']
var_arb = ['ARB', 'ENV', 'SOC', 'GOV']

data_ret = pd.DataFrame(columns = ['RET', 'ENV', 'SOC', 'GOV'])
data_arb = pd.DataFrame(columns = ['ARB', 'ENV', 'SOC', 'GOV'])

target_ret = 'RET'
target_arb = 'ARB'

ind = ['ENV', 'SOC', 'GOV']

for n in range(0, len(DATA)):

	ret = DATA['RET'][n]
	arb = DATA['ARB'][n]
	env = DATA['ENV'][n]
	soc = DATA['SOC'][n]
	gov = DATA['GOV'][n]

	data_ret.loc[n] = [ret, env, soc, gov]
	data_arb.loc[n] = [arb, env, soc, gov]

print('')
print('\n========================= Univariant Granger - Returns =========================')

print('')
print('\n=============== General ESG ===============')

result_esg = grangercausalitytests(DATA1[['RET', 'ESG']], max_lag, verbose=True)

print('')
print('\n=============== Environmental ESG ===============')

result_env = grangercausalitytests(DATA1[['RET', 'ENV']], max_lag, verbose=True)

print('')
print('\n=============== Social ESG ===============')
print('### Social ESG ###')

result_soc = grangercausalitytests(DATA1[['RET', 'SOC']], max_lag, verbose=True)

print('')
print('\n=============== Governance ESG ===============')

result_gov = grangercausalitytests(DATA1[['RET', 'GOV']], max_lag, verbose=True)

print('')
print('\n========================= Univariant Granger - Arbitrage =========================')

print('')
print('\n=============== General ESG ===============')

try:
	result_esg = grangercausalitytests(DATA[['ARB', 'ESG']], max_lag, verbose=True)
except:
	print('Results not available.')

print('')
print('\n=============== Environmental ESG ===============')

try:
	result_env = grangercausalitytests(DATA[['ARB', 'ENV']], max_lag, verbose=True)
except:
	print('Results not available.')

print('')
print('\n=============== Social ESG ===============')

try:
	result_soc = grangercausalitytests(DATA[['ARB', 'SOC']], max_lag, verbose=True)
except:
	print('Results not available.')

print('')
print('\n=============== Governance ESG ===============')

try:
	result_gov = grangercausalitytests(DATA[['ARB', 'GOV']], max_lag, verbose=True)
except:
	print('Results not available.')

##### ELASTIC NET #####

print('\n========================= Elastic Net - Returns =========================')

net(DATA1, Model = 1, n_bootstrap = 500)

print('\n========================= Elastic Net - Arbitrage =========================')

net(DATA, Model = 2, n_bootstrap = 500)

#################### END OF THE CODE ####################
