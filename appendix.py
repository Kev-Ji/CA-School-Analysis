import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import scipy.stats as stats

OUTDIR = "appendix_tables"
os.makedirs(OUTDIR, exist_ok=True)

def save_and_print(df, name, note=""):
    path = os.path.join(OUTDIR, f"{name}.csv")
    df.to_csv(path)
    print(f"\n{'='*70}\n{name}{'  — ' + note if note else ''}\n{'='*70}")
    print(df.to_string())
    print(f"[saved -> {path}]")

# ============================================================
# region DATA LOADING / CLEANING  (mirrors analysis.py "region data")
# ============================================================

final = pd.read_csv("data/final.csv")

final = final[
    ~final["District Type (District)"].isin(["County Office of Education (COE)"])
]

final = final.rename(columns={
    'Unnamed: 0': 'idx',
    'District Name': 'dist_name',
    'County Name (District)': 'county',
    'District Type (District)': 'dist_type',
    'Census Day Enrollment (District)': 'enrollment',
    'non_charter_math_caaspp': 'caaspp_math',
    'non_charter_ela_caaspp': 'caaspp_ela',
    'Student/Teacher Ratio (District)': 'stu_teach_ratio',
    'Free/Reduced Meals % (District)': 'pct_frpm',
    'English Learners % (District)': 'pct_el',
    'Ethnic Diversity Index (District)': 'diversity_idx',
    'SWDpct': 'pct_swd',
    'Locale [District] 2024-25': 'locale',
    'EnrollNonCharter': 'enroll_noncharter',
    'Pupil/Teacher Ratio [District] 2024-25': 'pupil_teach_ratio',
    'Teaching Days (District)': 'teaching_days',
    'Total Number Operational Schools [Public School] 2024-25': 'n_schools',
    'CDSCode': 'cds_code',
    'Agency Name Clean': 'agency_name_clean',
    'AssistStatus': 'assist_status',
    'Latitude [District] 2024-25': 'lat',
    'Longitude [District] 2024-25': 'lon',
    'median_income': 'median_income',
    'unemployment_pct': 'unemployment_pct',
    'poverty_pct': 'poverty_pct',
    'bach_pct': 'bach_pct',
    'DistrctAreaSqMi': 'area_sq_mi',
})

locale_map = {
    '11-City: Large': 'City', '12-City: Mid-size': 'City', '13-City: Small': 'City',
    '21-Suburb: Large': 'Suburb', '22-Suburb: Mid-size': 'Suburb', '23-Suburb: Small': 'Suburb',
    '31-Town: Fringe': 'Town', '32-Town: Distant': 'Town', '33-Town: Remote': 'Town',
    '41-Rural: Fringe': 'Rural', '42-Rural: Distant': 'Rural', '43-Rural: Remote': 'Rural',
}
assist_map = {
    'Differentiated, Year 1': 'Differentiated',
    'Differentiated, Year 2': 'Differentiated',
    'General': 'General',
}
dist_type_map = {
    'Elementary School District': 'Elementary',
    'High School District': 'High School',
    'Unified School District': 'Unified',
    'Union Elementary School District': 'Elementary',
    'Union High School District': 'High School',
}

final['assist_status'] = final['assist_status'].map(assist_map)
final['locale'] = final['locale'].map(locale_map)
final['dist_type'] = final['dist_type'].map(dist_type_map)

final['caaspp_ela'] = pd.to_numeric(final['caaspp_ela'], errors='coerce')
final['caaspp_math'] = pd.to_numeric(final['caaspp_math'], errors='coerce')
final['median_income'] = pd.to_numeric(final['median_income'], errors='coerce')
final['pupil_teach_ratio'] = pd.to_numeric(final['pupil_teach_ratio'], errors='coerce')
final['enroll_noncharter'] = np.log1p(final['enroll_noncharter'])
final = final[np.exp(final['enroll_noncharter']) > 30]

# endregion

# ============================================================
# region SPATIAL SETUP + BASELINE OLS  (Tables B1, B2, C1)
# ============================================================

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from libpysal.weights import KNN
from esda.moran import Moran

base_cols = ['caaspp_ela', 'caaspp_math', 'lat', 'lon', 'dist_name']
df_spatial = final.dropna(subset=base_cols).copy()

numeric_preds = [
    'pct_frpm', 'bach_pct', 'median_income', 'pct_el', 'diversity_idx',
    'pct_swd', 'unemployment_pct', 'pupil_teach_ratio', 'teaching_days',
]


existing_numeric = [c for c in numeric_preds if c in df_spatial.columns]

imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
df_spatial[existing_numeric] = imputer.fit_transform(df_spatial[existing_numeric])

for cat_col in ['locale', 'dist_type']:
    if cat_col in df_spatial.columns:
        df_spatial[cat_col] = df_spatial[cat_col].fillna(df_spatial[cat_col].mode()[0])

coords = np.array(list(zip(df_spatial['lon'], df_spatial['lat'])))
w = KNN.from_array(coords, k=8)
w_dist = KNN.from_array(coords, k=8, distance_metric='euclidean')
for i, neighbors in w_dist.neighbors.items():
    distances = [w_dist.weights[i][j] for j in range(len(neighbors))]
    w.weights[i] = [1.0 / (d + 0.0001) for d in distances]
w.transform = 'r'

formula = (
    '{outcome} ~ pct_frpm '
    '+ bach_pct + median_income + pct_el + diversity_idx '
    '+ locale + pct_swd + enroll_noncharter + dist_type + teaching_days + unemployment_pct + enroll_noncharter'
)

def clean_index(idx):
    if 'locale[T.' in idx:
        return idx.replace('locale[T.', '').replace(']', '') + ' locale'
    if idx == 'Intercept':
        return 'Intercept'
    return idx

moran_rows = []
ols_results = {}

for outcome, label in [('caaspp_ela', 'CAASPP ELA'), ('caaspp_math', 'CAASPP Math')]:
    res = smf.ols(formula.format(outcome=outcome), data=df_spatial).fit()
    df_spatial[f'ols_resid_{outcome}'] = res.resid.values
    ols_results[outcome] = res

    table = pd.DataFrame({
        'Coefficient': res.params,
        'Std. Error': res.bse,
        't-stat': res.tvalues,
        'p-value': res.pvalues,
    }).round(3)
    table.index = [clean_index(i) for i in table.index]

    save_and_print(
        table, f"Table_B{'1' if outcome == 'caaspp_ela' else '2'}_{label.replace(' ', '_')}",
        note=f"Observations: {int(res.nobs)} | R²: {res.rsquared:.3f} | Adj. R²: {res.rsquared_adj:.3f}"
    )

    moran = Moran(res.resid.values, w)
    moran_rows.append({
        'Outcome': label, 'OLS R²': round(res.rsquared, 4),
        "Moran's I": round(moran.I, 4), 'p': round(moran.p_sim, 4), 'z': round(moran.z_norm, 4),
    })

moran_table = pd.DataFrame(moran_rows).set_index('Outcome')
save_and_print(moran_table, "Table_C1_Morans_I", note="IDW k=8 nearest neighbors, baseline OLS residuals")

# endregion

# ============================================================
# region OLS + RANDOM FOREST + ELASTIC NET  (Table B3)
# ============================================================

from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

MODEL_FEATURES = [
    'median_income', 'bach_pct', 'unemployment_pct', 'poverty_pct', 'locale',
    'pct_frpm', 'pct_el', 'pct_swd', 'diversity_idx', 'enroll_noncharter','dist_type', 'teaching_days'
]
MODEL_FEATURES = [c for c in MODEL_FEATURES if c in df_spatial.columns]

cv = KFold(n_splits=5, shuffle=True, random_state=42)
rf_base = RandomForestRegressor(n_estimators=1000, max_features='sqrt', random_state=42, n_jobs=-1)

outcomes = ['caaspp_ela', 'caaspp_math']
ols_resids = {}
rf_resids = {}
enet_resids = {}
r2_rows = []

for outcome in outcomes:
    model_data = df_spatial[MODEL_FEATURES + [outcome, 'dist_name']].copy().dropna(subset=[outcome])

    cat_cols = [c for c in ['locale', 'dist_type'] if c in model_data.columns]
    if cat_cols:
        model_data = pd.get_dummies(model_data, columns=cat_cols, dtype=float, drop_first=True)

    feature_cols = [c for c in model_data.columns if c not in [outcome, 'dist_name']]
    X = model_data[feature_cols]
    y = model_data[outcome]
    num_cols = X.select_dtypes(include='number').columns

    preprocess = ColumnTransformer(
        [('iterative', IterativeImputer(estimator=BayesianRidge(), random_state=42, max_iter=10), num_cols)],
        remainder='passthrough'
    )

    # --- OLS ---
    ols_pipe = Pipeline([
        ('prep', preprocess),
        ('ols', LinearRegression())
    ])
    ols_pred = cross_val_predict(ols_pipe, X, y, cv=cv)
    ols_r2 = r2_score(y, ols_pred)
    ols_resids[outcome] = pd.Series(y.values - ols_pred, index=model_data.index, name=f'ols_resid_{outcome}')

    # --- Elastic Net ---
    enet_pipe = Pipeline([
        ('prep', preprocess),
        ('enet', ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1], cv=5, random_state=42, max_iter=10000))
    ])
    enet_pred = cross_val_predict(enet_pipe, X, y, cv=cv)
    enet_r2 = r2_score(y, enet_pred)
    enet_resids[outcome] = pd.Series(y.values - enet_pred, index=model_data.index, name=f'enet_resid_{outcome}')

    # --- Random Forest ---
    rf_pipe = Pipeline([('prep', preprocess), ('rf', rf_base)])
    rf_pred = cross_val_predict(rf_pipe, X, y, cv=cv)
    rf_r2 = r2_score(y, rf_pred)
    rf_resids[outcome] = pd.Series(y.values - rf_pred, index=model_data.index, name=f'rf_resid_{outcome}')

    r2_rows.append({
        'Outcome': outcome,
        'OLS CV R²': round(ols_r2, 4),
        'Elastic Net CV R²': round(enet_r2, 4),
        'Random Forest CV R²': round(rf_r2, 4)
    })

r2_table = pd.DataFrame(r2_rows).set_index('Outcome')
save_and_print(r2_table, "Table_B3_Model_R2", note="Five-fold CV out-of-sample R², single shared feature set (Appendix B)")

# endregion

# ============================================================
# region WINSORIZED PCA  (six components: OLS, elastic net, random forest)
# ============================================================

from sklearn.decomposition import PCA
from scipy.stats import zscore

combined = df_spatial[['dist_name', 'county']].copy()
for outcome in outcomes:
    combined[f'ols_resid_{outcome}'] = df_spatial[f'ols_resid_{outcome}'].values
    combined = combined.join(enet_resids[outcome], how='left')
    combined = combined.join(rf_resids[outcome], how='left')

combined['enroll_noncharter_raw'] = np.exp(
    df_spatial.loc[df_spatial['dist_name'].isin(combined['dist_name']), 'enroll_noncharter'].values
)
MIN_ENROLLMENT = 30
combined = combined[combined['enroll_noncharter_raw'] >= MIN_ENROLLMENT].copy()
combined = combined.merge(df_spatial[['dist_name', 'lat', 'lon']], on='dist_name', how='left')
combined = combined.merge(df_spatial[['dist_name', 'locale']], on='dist_name', how='left')

components = ['ols_ela_z', 'ols_math_z', 'enet_ela_z', 'enet_math_z', 'rf_ela_z', 'rf_math_z']
for outcome, prefix in [('caaspp_ela', 'ela'), ('caaspp_math', 'math')]:
    combined[f'ols_{prefix}_z'] = zscore(combined[f'ols_resid_{outcome}'], nan_policy='omit')
    combined[f'enet_{prefix}_z'] = zscore(combined[f'enet_resid_{outcome}'], nan_policy='omit')
    combined[f'rf_{prefix}_z'] = zscore(combined[f'rf_resid_{outcome}'], nan_policy='omit')

for col in components:
    lower, upper = combined[col].quantile(0.01), combined[col].quantile(0.99)
    combined[col] = combined[col].clip(lower=lower, upper=upper)

pca = PCA(n_components=1)
pca_raw = pca.fit_transform(combined[components]).ravel()
combined["pca_raw"] = pca_raw

if combined[components[0]].corr(combined["pca_raw"]) < 0:
    combined["pca_raw"] *= -1

print(f"\nPCA component 1 variance explained: {pca.explained_variance_ratio_[0]*100:.1f}%")

# ---------------------------------------------------------
# Bootstrap PCA uncertainty
# ---------------------------------------------------------

from sklearn.utils import resample

B = 1000
rng = 42

X = combined[components].values
N = len(combined)

boot_scores = np.empty((B, N))

ref_loading = pca.components_[0].copy()

for b in range(B):

    idx = resample(
        np.arange(N),
        replace=True,
        n_samples=N,
        random_state=rng + b
    )

    X_boot = X[idx]

    pca_boot = PCA(n_components=1)
    pca_boot.fit(X_boot)

    if np.dot(pca_boot.components_[0], ref_loading) < 0:
        pca_boot.components_[0] *= -1

    boot_scores[b] = pca_boot.transform(X).ravel()

combined["pc1_var"] = boot_scores.var(axis=0, ddof=1)
combined["pc1_se"] = np.sqrt(combined["pc1_var"])

print(
    f"Bootstrap PC1 SE "
    f"(median={combined['pc1_se'].median():.4f}, "
    f"max={combined['pc1_se'].max():.4f})"
)

# endregion

# endregion

# ============================================================
# region EMPIRICAL BAYES SHRINKAGE OF COMPOSITE SCORE (Table D1)
# ============================================================

KAPPA = 0.15


def compute_eb_shrinkage(kappa, record_rows=False, store_columns=False):
    """Run the locale-wise EB shrinkage for a given noise share kappa.

    Returns (eb_shrunken_raw: pd.Series aligned to combined.index,
    eb_rows: list of per-locale summary dicts, or [] if record_rows
    is False). If store_columns is True, also writes shrinkage_weight,
    sampling_var, and posterior_var back onto `combined` (only
    meaningful for the single kappa actually used for Table D1).
    """
    shrunken = pd.Series(index=combined.index, dtype=float)
    rows = []

    if store_columns:
        combined["shrinkage_weight"] = np.nan
        combined["sampling_var"] = np.nan
        combined["posterior_var"] = np.nan

    for locale_name, group in combined.groupby("locale"):

        y = group["pca_raw"].values
        n = group["enroll_noncharter_raw"].values
        pc1_var = group["pc1_var"].values

        y_bar = np.average(y, weights=n)

        sample_var = np.average((y - y_bar) ** 2, weights=n)
        mean_inv_n = np.average(1.0 / n, weights=n)

        if len(y) < 3:
            shrunken.loc[group.index] = y
            if store_columns:
                combined.loc[group.index, "shrinkage_weight"] = 1.0
                combined.loc[group.index, "sampling_var"] = np.nan
                combined.loc[group.index, "posterior_var"] = pc1_var
            if record_rows:
                rows.append({
                    "Locale": locale_name,
                    "Districts": len(group),
                    "Min weight": 1.0,
                    "Median weight": 1.0,
                    "Mean weight": 1.0,
                    "Max weight": 1.0,
                    "Median PC1 SE": group["pc1_se"].median(),
                    "Mean PC1 SE": group["pc1_se"].mean(),
                })
            continue

        sigma_sq = kappa * sample_var / np.median(1.0 / n)
        tau_sq = max(0.0, sample_var - sigma_sq * mean_inv_n)

        sampling_var = sigma_sq / n
        posterior_var = sampling_var + pc1_var

        w_i = tau_sq / (tau_sq + posterior_var)

        shrunken.loc[group.index] = y_bar + w_i * (y - y_bar)

        if store_columns:
            combined.loc[group.index, "shrinkage_weight"] = w_i
            combined.loc[group.index, "sampling_var"] = sampling_var
            combined.loc[group.index, "posterior_var"] = posterior_var

        if record_rows:
            rows.append({
                "Locale": locale_name,
                "Districts": len(group),
                "Min weight": w_i.min(),
                "Median weight": np.median(w_i),
                "Mean weight": np.mean(w_i),
                "Max weight": w_i.max(),
                "Median PC1 SE": group["pc1_se"].median(),
                "Mean PC1 SE": group["pc1_se"].mean(),
            })

    return shrunken, rows


combined["eb_shrunken_raw"], eb_rows = compute_eb_shrinkage(
    KAPPA, record_rows=True, store_columns=True
)

eb_table = (
    pd.DataFrame(eb_rows)
      .set_index("Locale")
      .round(4)
)

save_and_print(
    eb_table,
    "Table_D1_Empirical_Bayes",
    note=(
        "Empirical Bayes shrinkage weights applied to the composite PCA score. "
        f"Noise share (kappa = {KAPPA}), fixed rather than independently "
        "estimated -- see Table D2 for a sensitivity sweep. Observation-level "
        "variance is defined as the sum of enrollment-based sampling variance "
        "(kappa-scaled sigma^2/n) and bootstrap-estimated PCA variance."
    ),
)

# --- Sensitivity sweep over kappa (Table D2) ---


kappa_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
baseline_shrunk, _ = compute_eb_shrinkage(KAPPA, record_rows=False)

sensitivity_rows = []
for k in kappa_grid:
    shrunk_k, rows_k = compute_eb_shrinkage(k, record_rows=True)
    mean_weight = np.mean([r["Mean weight"] for r in rows_k if r["Districts"] >= 3])
    rank_corr = stats.spearmanr(
        baseline_shrunk, shrunk_k, nan_policy="omit"
    ).correlation
    max_abs_diff = (shrunk_k - baseline_shrunk).abs().max()
    sensitivity_rows.append({
        "kappa": k,
        "Mean shrinkage weight": round(mean_weight, 4),
        "Spearman rank corr. vs. kappa=0.15": round(rank_corr, 4),
        "Max |score change| vs. kappa=0.15": round(max_abs_diff, 4),
    })

sensitivity_table = pd.DataFrame(sensitivity_rows).set_index("kappa")

save_and_print(
    sensitivity_table,
    "Table_D2_Kappa_Sensitivity",
    note=(
        "Sensitivity of Empirical Bayes shrinkage to the noise-share "
        "parameter kappa (Table D1 uses kappa = 0.15). District rankings are "
        "compared to the kappa = 0.15 baseline via Spearman rank correlation "
        "on the shrunken composite score, prior to spatial smoothing."
    ),
)

# endregion
# endregion

# ============================================================
# region SAR SMOOTHING OF COMPOSITE SCORE  (Table C2)
# ============================================================

import spreg
from libpysal.weights import lag_spatial
from scipy.spatial.distance import euclidean

coords_sub = combined[['lon', 'lat']].values
w_sub = KNN.from_array(coords_sub, k=8)
for i, neighbors in w_sub.neighbors.items():
    distances = [euclidean(coords_sub[i], coords_sub[j]) for j in neighbors]
    w_sub.weights[i] = [1.0 / (d + 0.0001) for d in distances]
w_sub.transform = 'r'

X_intercept = np.ones((len(combined), 1))
y_sar = combined['eb_shrunken_raw'].values.reshape(-1, 1)
sar_model = spreg.ML_Lag(y_sar, X_intercept, w=w_sub, name_y='eb_shrunken_raw')
rho = max(0.0, min(sar_model.rho, 0.5))
neighbor_avg = lag_spatial(w_sub, combined['eb_shrunken_raw'].values)
combined['sar_smoothed'] = (1 - rho) * combined['eb_shrunken_raw'] + rho * neighbor_avg

sar_table = pd.DataFrame([{
    'Component': 'Composite (post-shrinkage) score', 'rho': round(rho, 3),
    'Own district': f"{(1-rho)*100:.1f}%", 'Neighbor average': f"{rho*100:.1f}%",
}]).set_index('Component')
save_and_print(sar_table, "Table_C2_SAR_Blending")

combined['pca_z'] = zscore(combined['sar_smoothed'], nan_policy='omit')
final_lower, final_upper = combined['pca_z'].quantile(0.01), combined['pca_z'].quantile(0.99)
combined['pca_z'] = combined['pca_z'].clip(lower=final_lower, upper=final_upper)

# endregion

# ============================================================
# region EXTERNAL VALIDATION  (Table G1)
# ============================================================

try:
    try:
        awards = pd.read_csv("data/2024-25award.csv")
    except FileNotFoundError:
        awards = pd.read_csv("data/dsaawards.xlsx - CA Distinguished Schools.csv")

    recent_awards = awards[awards['Year'].isin([2024, 2025])].copy()
    if recent_awards.empty:
        recent_awards = awards[awards['Year'] == awards['Year'].max()].copy()

    school_counts = final[['dist_name', 'cds_code', 'agency_name_clean', 'n_schools']].copy()
    award_cds_col = next((c for c in recent_awards.columns if 'cds' in c.lower()), None)

    if award_cds_col:
        recent_awards['merge_target'] = recent_awards[award_cds_col].astype(str).str.zfill(14)
        school_counts['merge_target'] = school_counts['cds_code'].astype(str).str.zfill(14)
    else:
        dist_col = 'District' if 'District' in recent_awards.columns else 'District Name'
        recent_awards['merge_target'] = recent_awards[dist_col].astype(str).str.upper().str.strip()
        school_counts['merge_target'] = school_counts['agency_name_clean'].astype(str).str.upper().str.strip()

    award_counts = recent_awards.groupby('merge_target').size().reset_index(name='award_count')
    val_df = school_counts.merge(award_counts[['merge_target', 'award_count']], on='merge_target', how='left')
    val_df['award_count'] = val_df['award_count'].fillna(0)
    val_df['n_schools'] = pd.to_numeric(val_df['n_schools'], errors='coerce').fillna(1)
    val_df['n_schools'] = val_df['n_schools'].apply(lambda x: x if x > 0 else 1)

    raw_rates = val_df['award_count'] / val_df['n_schools']
    global_mean = raw_rates.mean()
    weighted_var = np.average((raw_rates - global_mean) ** 2, weights=val_df['n_schools'])

    if weighted_var > 0 and weighted_var < (global_mean * (1 - global_mean)):
        gamma = (global_mean * (1 - global_mean) / weighted_var) - 1
        alpha = global_mean * gamma
        beta = (1 - global_mean) * gamma
    else:
        prior_weight = val_df['n_schools'].median()
        alpha = global_mean * prior_weight
        beta = (1 - global_mean) * prior_weight

    val_df['cds_density_smoothed'] = (val_df['award_count'] + alpha) / (val_df['n_schools'] + alpha + beta)

    val_df_final = combined.merge(
        val_df[['dist_name', 'cds_density_smoothed', 'award_count']], on='dist_name', how='left'
    )
    val_df_final['cds_density_smoothed'] = val_df_final['cds_density_smoothed'].fillna(alpha / (alpha + beta))
    val_df_final['award_count'] = val_df_final['award_count'].fillna(0)

    tau, p_val = stats.kendalltau(
        val_df_final['pca_z'], val_df_final['cds_density_smoothed'], nan_policy='omit'
    )

    a8 = pd.DataFrame({
        'Metric': ['Empirical Bayes prior (alpha)', 'Empirical Bayes prior (beta)',
                   'Awards mapped (2024-2025)', "Kendall's tau-b (tie-adjusted)", 'p-value'],
        'Value': [round(alpha, 3), round(beta, 3), int(val_df['award_count'].sum()),
                  round(tau, 4), p_val],
    }).set_index('Metric')

    save_and_print(a8, "Table_G1_External_Validation")

except Exception as e:
    print(f"\n[Table G1 FAILED] {e}")
    print("Check that the award CSV path/columns match your data directory.")

# endregion

# ============================================================
# region TRENDS ACROSS DISTRICTS
# ============================================================

combined['final_z_standardized'] = combined['pca_z']


def assign_tier(z):
    if z >= 1.5:
        return "Significant Overperformer"
    elif z >= 0.5:
        return "Moderate Overperformer"
    elif z > -0.5:
        return "Expected Performer"
    elif z > -1.5:
        return "Moderate Underperformer"
    else:
        return "Significant Underperformer"


combined['performance_tier'] = combined['final_z_standardized'].apply(assign_tier)

print("\n" + "="*50)
print("TIER ASSIGNMENT: CATEGORIZING PERFORMANCE")
print("="*50)
print(combined['performance_tier'].value_counts())

# --- Demographic comparison: overperformers vs. underperformers ---

demo = final.merge(
    combined[['dist_name', 'performance_tier']], on='dist_name', how='inner'
)

overperformers = demo[demo['performance_tier'].isin(
    ['Moderate Overperformer', 'Significant Overperformer']
)]
underperformers = demo[demo['performance_tier'].isin(
    ['Moderate Underperformer', 'Significant Underperformer']
)]


def summarize_demo(group):
    return {
        'Rural Districts (%)': 100 * group['locale'].eq('Rural').mean(),
        'Average Enrollment': group['enrollment'].mean(),
        'Ethnic Diversity Index': group['diversity_idx'].mean(),
        'English Learners (%)': group['pct_el'].mean(),
        'Free/Reduced Meals (%)': group['pct_frpm'].mean(),
        "Adults with Bachelor's Degree (%)": group['bach_pct'].mean(),
    }


demo_table = pd.DataFrame({
    f'Overperformers (n={len(overperformers)})': summarize_demo(overperformers),
    f'Underperformers (n={len(underperformers)})': summarize_demo(underperformers),
})
demo_table.loc['Average Enrollment'] = demo_table.loc['Average Enrollment'].round(0)
demo_table = demo_table.round(1)

save_and_print(
    demo_table,
    "Table_2_Demographic_Comparison",
    note="Overperformers (Moderate + Significant) vs. Underperformers (Moderate + Significant)",
)

# --- Demographic comparison: significant tier only ---

sig_overperformers = demo[demo['performance_tier'] == 'Significant Overperformer']
sig_underperformers = demo[demo['performance_tier'] == 'Significant Underperformer']

sig_demo_table = pd.DataFrame({
    f'Significant Overperformers (n={len(sig_overperformers)})': summarize_demo(sig_overperformers),
    f'Significant Underperformers (n={len(sig_underperformers)})': summarize_demo(sig_underperformers),
})
sig_demo_table.loc['Average Enrollment'] = sig_demo_table.loc['Average Enrollment'].round(0)
sig_demo_table = sig_demo_table.round(1)

save_and_print(
    sig_demo_table,
    "Table_3_Significant_Tier_Demographic_Comparison",
    note="Significant Overperformers vs. Significant Underperformers only",
)

# --- Differentiated Assistance status by performance tier ---

assistance_df = final.merge(
    combined[['dist_name', 'performance_tier']], on='dist_name', how='inner'
)
assistance_df['assistance_type'] = np.where(
    assistance_df['assist_status'].str.contains('Differentiated', case=False, na=False),
    'Differentiated',
    'General',
)

tier_order = [
    'Moderate Underperformer',
    'Expected Performer',
    'Moderate Overperformer',
    'Significant Underperformer',
    'Significant Overperformer',
]

assistance_table = (
    pd.crosstab(assistance_df['performance_tier'], assistance_df['assistance_type'])
      .reindex(tier_order)
      .fillna(0)
      .astype(int)
)
assistance_table['Total'] = assistance_table.sum(axis=1)
assistance_table['% Differentiated'] = (
    assistance_table['Differentiated'] / assistance_table['Total'] * 100
)
assistance_table = assistance_table[['Differentiated', '% Differentiated', 'General', 'Total']]

total_diff = assistance_table['Differentiated'].sum()
total_gen = assistance_table['General'].sum()
total_n = assistance_table['Total'].sum()
assistance_table.loc['Total'] = [total_diff, 100 * total_diff / total_n, total_gen, total_n]
assistance_table['% Differentiated'] = assistance_table['% Differentiated'].round(1)

save_and_print(
    assistance_table,
    "Table_4_Assistance_By_Tier",
    note="District assistance status by performance tier; rows ordered by % receiving Differentiated Assistance",
)

# endregion

print(f"\nAll tables written to ./{OUTDIR}/")