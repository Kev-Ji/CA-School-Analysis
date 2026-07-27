import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# region data

final = pd.read_csv("data/final.csv")

final = final[
    ~final["District Type (District)"].isin([
        "County Office of Education (COE)"
    ])
]

final = final.rename(columns={
    'Unnamed: 0': 'idx',
    'District Name': 'dist_name',
    'County Name (District)': 'county',
    'District Type (District)': 'dist_type',
    'All Charter Schools (Y/N) (District)': 'all_charter',
    'Grade Span (District)': 'grade_span',
    'Zip (District)': 'zip',
    'City (District)': 'city',
    'Census Day Enrollment (District)': 'enrollment',
    'English Learners % (District)': 'pct_el',
    'American Indian or Alaska Native % (District)': 'pct_aian',
    'Asian % (District)': 'pct_asian',
    'Black or African American % (District)': 'pct_black',
    'Filipino % (District)': 'pct_filipino',
    'Hispanic or Latino % (District)': 'pct_hispanic',
    'Native Hawaiian or Pac Islander % (District)': 'pct_nhpi',
    'Two or More Races % (District)': 'pct_multi',
    'None Reported % (District)': 'pct_none_reported',
    'White % (District)': 'pct_white',
    'Free/Reduced Meals % (District)': 'pct_frpm',
    'Fluent English Proficient (FEP) % (District)': 'pct_fep',
    'Ethnic Diversity Index (District)': 'diversity_idx',
    'Cohort Graduates % (District)': 'pct_grad',
    'Cohort Grads by Socio. Econ. DisAdvtg % (District)': 'pct_grad_sed',
    'non_charter_math_caaspp': 'caaspp_math',
    'non_charter_ela_caaspp': 'caaspp_ela',
    'Student/Teacher Ratio (District)': 'stu_teach_ratio',
    '1st Year Teachers (District)': 'pct_first_yr_teachers',
    'Experienced Teachers (District)': 'pct_exp_teachers',
    'Teacher Service Days (District)': 'teacher_svc_days',
    'Teaching Days (District)': 'teaching_days',
    'Inexperienced Teachers (District)': 'pct_inexp_teachers',
    'Avg Years Teaching (District)': 'avg_yrs_teaching',
    'Suspension Rate (District)': 'suspension_rate',
    'Chronic Absenteeism % (District)': 'pct_chronic_absent',
    'Current Exp of Educ per ADA (Ed Code 41372) (District)': 'exp_per_ada',
    'Gen Fund Exp by Activity - 1000-1999 Instruction Exp % (District)': 'pct_exp_instruction',
    'Gen Fund Exp by Activity - 2000-2999 Instruc-related Svcs Exp % (District)': 'pct_exp_instruct_svc',
    'Gen Fund Exp by Activity - 3000-3999 Pupil Services Exp % (District)': 'pct_exp_pupil_svc',
    'Gen Fund Exp by Activity - 4000-4999 Ancillary Services Exp % (District)': 'pct_exp_ancillary',
    'Gen Fund Exp by Activity - 6000-6999 Enterprise Exp % (District)': 'pct_exp_enterprise',
    'Gen Fund Exp by Activity - 7000-7999 General Administration Exp % (District)': 'pct_exp_admin',
    'Gen Fund Exp by Activity - 8000-8999 Plant Services Exp % (District)': 'pct_exp_plant',
    'District Name Clean': 'dist_name_clean',
    'Agency Name': 'agency_name',
    'Agency Name Clean': 'agency_name_clean',
    'Agency Name [District] 2024-25': 'agency_name_2425',
    'Agency ID - NCES Assigned [District] Latest available year': 'nces_id',
    'State Agency ID [District] 2024-25': 'state_agency_id',
    'State Name [District] Latest available year': 'state_name',
    'State Name [District] 2024-25': 'state_name_2425',
    'State Abbr [District] Latest available year': 'state_abbr',
    'ANSI/FIPS State Code [District] Latest available year': 'fips_code',
    'County Name [District] 2024-25': 'county_2425',
    'County Number [District] 2024-25': 'county_num',
    'Latitude [District] 2024-25': 'lat',
    'Longitude [District] 2024-25': 'lon',
    'Location ZIP [District] 2024-25': 'zip_2425',
    'Location ZIP4 [District] 2024-25': 'zip4_2425',
    'Location City [District] 2024-25': 'city_2425',
    'GeoId': 'geo_id',
    'Geography': 'geography',
    'LEAID': 'leaid',
    'Locale [District] 2024-25': 'locale',
    'Agency Type [District] 2024-25': 'agency_type',
    'Agency Level (SY 2017-18 onward) [District] 2024-25': 'agency_level',
    'LEA Charter Status [District] 2024-25': 'lea_charter_status',
    'Start of Year Status [District] 2024-25': 'status_start',
    'Updated Status [District] 2024-25': 'status_updated',
    'Effective Date of Updated Status [District] 2024-25': 'status_eff_date',
    'Web Site URL [District] 2024-25': 'website',
    'CBSA Name [District] 2024-25': 'cbsa_name',
    'CBSA ID [District] 2024-25': 'cbsa_id',
    'CSA Name [District] 2024-25': 'csa_name',
    'CSA ID [District] 2024-25': 'csa_id',
    'Metro Micro Area Code [District] 2024-25': 'metro_code',
    'Congressional Code [District] 2024-25': 'congressional_code',
    'Supervisory Union (ID) Number [District] 2024-25': 'supervisory_union_id',
    'Lowest Grade Offered [District] 2024-25': 'grade_low',
    'Highest Grade Offered [District] 2024-25': 'grade_high',
    'Total Number Operational Schools [Public School] 2024-25': 'n_schools',
    'Total Number Operational Charter Schools [Public School] 2024-25': 'n_charter_schools',
    'Total Students All Grades (Excludes AE) [District] 2024-25': 'n_students',
    'Total Students All Grades (Includes AE) [District] 2024-25': 'n_students_ae',
    'Male Students [District] 2024-25': 'n_male',
    'Female Students [District] 2024-25': 'n_female',
    'Total Race/Ethnicity [District] 2024-25': 'n_race_total',
    'American Indian/Alaska Native Students [District] 2024-25': 'n_aian',
    'Asian or Asian/Pacific Islander Students [District] 2024-25': 'n_asian',
    'Hispanic Students [District] 2024-25': 'n_hispanic',
    'Black or African American Students [District] 2024-25': 'n_black',
    'White Students [District] 2024-25': 'n_white',
    'Nat. Hawaiian or Other Pacific Isl. Students [District] 2024-25': 'n_nhpi',
    'Two or More Races Students [District] 2024-25': 'n_multi',
    'American Indian/Alaska Native - male [District] 2024-25': 'n_aian_m',
    'American Indian/Alaska Native - female [District] 2024-25': 'n_aian_f',
    'Asian or Asian/Pacific Islander - male [District] 2024-25': 'n_asian_m',
    'Asian or Asian/Pacific Islander - female [District] 2024-25': 'n_asian_f',
    'Hispanic - male [District] 2024-25': 'n_hispanic_m',
    'Hispanic - female [District] 2024-25': 'n_hispanic_f',
    'Black or African American - male [District] 2024-25': 'n_black_m',
    'Black or African American - female [District] 2024-25': 'n_black_f',
    'White - male [District] 2024-25': 'n_white_m',
    'White - female [District] 2024-25': 'n_white_f',
    'Nat. Hawaiian or Other Pacific Isl. - male [District] 2024-25': 'n_nhpi_m',
    'Nat. Hawaiian or Other Pacific Isl. - female [District] 2024-25': 'n_nhpi_f',
    'Two or More Races - male [District] 2024-25': 'n_multi_m',
    'Two or More Races - female [District] 2024-25': 'n_multi_f',
    'Full-Time Equivalent (FTE) Teachers [District] 2024-25': 'fte_teachers',
    'Pupil/Teacher Ratio [District] 2024-25': 'pupil_teach_ratio',
    'Total Staff [District] 2024-25': 'n_staff',
    'Paraprofessionals/Instructional Aides [District] 2024-25': 'n_paraprofessionals',
    'Instructional Coordinators [District] 2024-25': 'n_instruct_coord',
    'Elementary School Counselor [District] 2024-25': 'n_elem_counselor',
    'Secondary School Counselor [District] 2024-25': 'n_sec_counselor',
    'Other Guidance Counselors [District] 2024-25': 'n_other_counselor',
    'Total Guidance Counselors [District] 2024-25': 'n_counselors',
    'Librarians/media specialists [District] 2024-25': 'n_librarians',
    'Media Support Staff [District] 2024-25': 'n_media_staff',
    'LEA Administrators [District] 2024-25': 'n_lea_admin',
    'LEA Administrative Support Staff [District] 2024-25': 'n_lea_admin_support',
    'School Administrators [District] 2024-25': 'n_school_admin',
    'School Administrative Support Staff [District] 2024-25': 'n_school_admin_support',
    'Student Support Services Staff (w/o Psychology) [District] 2024-25': 'n_student_support',
    'School Psychologist [District] 2024-25': 'n_psychologist',
    'Other Support Services Staff [District] 2024-25': 'n_other_support',
    'median_income': 'median_income',
    'median_income_moe': 'income_moe',
    'unemployment_pct': 'unemployment_pct',
    'poverty_pct': 'poverty_pct',
    'bach_pct': 'bach_pct',
    'OBJECTID': 'objectid',
    'Year': 'year',
    'FedID': 'fed_id',
    'CDCode': 'cd_code',
    'CDSCode': 'cds_code',
    'CountyName': 'county_name',
    'DistrictName': 'district_name',
    'DistrictType': 'district_type',
    'GradeLow': 'grade_low_ca',
    'GradeHigh': 'grade_high_ca',
    'GradeLowCensus': 'grade_low_census',
    'GradeHighCensus': 'grade_high_census',
    'AssistStatus': 'assist_status',
    'CongressUS': 'congress_us',
    'SenateCA': 'senate_ca',
    'AssemblyCA': 'assembly_ca',
    'UpdateNotes': 'update_notes',
    'EnrollTotal': 'enroll_total',
    'EnrollCharter': 'enroll_charter',
    'EnrollNonCharter': 'enroll_noncharter',
    'AAcount': 'n_aa',
    'AApct': 'pct_aa',
    'AIcount': 'n_ai',
    'AIpct': 'pct_ai',
    'AScount': 'n_as',
    'ASpct': 'pct_as',
    'FIcount': 'n_fi',
    'FIpct': 'pct_fi',
    'HIcount': 'n_hi',
    'HIpct': 'pct_hi',
    'PIcount': 'n_pi',
    'PIpct': 'pct_pi',
    'WHcount': 'n_wh',
    'WHpct': 'pct_wh',
    'MRcount': 'n_mr',
    'MRpct': 'pct_mr',
    'NRcount': 'n_nr',
    'NRpct': 'pct_nr',
    'ELcount': 'n_el',
    'ELpct': 'pct_el_ca',
    'FOScount': 'n_fos',
    'FOSpct': 'pct_fos',
    'HOMcount': 'n_hom',
    'HOMpct': 'pct_hom',
    'MIGcount': 'n_mig',
    'MIGpct': 'pct_mig',
    'SWDcount': 'n_swd',
    'SWDpct': 'pct_swd',
    'SEDcount': 'n_sed',
    'SEDpct': 'sed_pct',
    'DistrctAreaSqMi': 'area_sq_mi',
    'Shape__Area': 'shape_area',
    'Shape__Length': 'shape_length',
})

locale_map = {
    '11-City: Large':      'City',
    '12-City: Mid-size':   'City',
    '13-City: Small':      'City',
    '21-Suburb: Large':    'Suburb',
    '22-Suburb: Mid-size': 'Suburb',
    '23-Suburb: Small':    'Suburb',
    '31-Town: Fringe':     'Town',
    '32-Town: Distant':    'Town',
    '33-Town: Remote':     'Town',
    '41-Rural: Fringe':    'Rural',
    '42-Rural: Distant':   'Rural',
    '43-Rural: Remote':    'Rural',
}

dist_type_map = {
    'Elementary School District':        'Elementary',
    'High School District':              'High School',
    'Unified School District':           'Unified',
    'Union Elementary School District':  'Elementary',
    'Union High School District':        'High School',
}

assist_map = {
    'Differentiated, Year 1': 'Differentiated',
    'Differentiated, Year 2': 'Differentiated',
    'General':                 'General'
}

final['assist_status'] = final['assist_status'].map(assist_map)
final['locale']        = final['locale'].map(locale_map)
final['dist_type']     = final['dist_type'].map(dist_type_map)

final['caaspp_ela']        = pd.to_numeric(final['caaspp_ela'],        errors='coerce')
final['caaspp_math']       = pd.to_numeric(final['caaspp_math'],       errors='coerce')
final['median_income']     = pd.to_numeric(final['median_income'],     errors='coerce')
final['pupil_teach_ratio'] = pd.to_numeric(final['pupil_teach_ratio'], errors='coerce')
final['fte_teachers']      = pd.to_numeric(final['fte_teachers'],      errors='coerce')
final['enroll_noncharter']      = np.log1p(final['enroll_noncharter'])

final = final[np.exp(final['enroll_noncharter']) > 30]

# endregion

# region spatial

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from libpysal.weights import KNN
from sklearn.metrics import r2_score
from esda.moran import Moran
base_cols = ['caaspp_ela', 'caaspp_math', 'lat', 'lon', 'dist_name']
df_spatial = final.dropna(subset=base_cols).copy()

numeric_preds = [
     'pct_frpm', 'bach_pct', 'median_income', 'pct_el', 'diversity_idx', 
    'pct_swd', 'enroll_total', 'unemployment_pct', 
    'pupil_teach_ratio', 'teaching_days', 'exp_per_ada', 'pct_exp_admin', 
    'pct_exp_pupil_svc'
]

existing_numeric = [col for col in numeric_preds if col in df_spatial.columns]

imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
df_spatial[existing_numeric] = imputer.fit_transform(df_spatial[existing_numeric])

for cat_col in ['locale', 'dist_type']:
    if cat_col in df_spatial.columns:
        df_spatial[cat_col] = df_spatial[cat_col].fillna(df_spatial[cat_col].mode()[0])
    
df_spatial['dist_name'] = df_spatial['dist_name'].values

coords = np.array(list(zip(df_spatial['lon'], df_spatial['lat'])))
from scipy.spatial.distance import euclidean

w = KNN.from_array(coords, k=8)

for i, neighbors in w.neighbors.items():
    distances = [euclidean(coords[i], coords[j]) for j in neighbors]
    w.weights[i] = [1.0 / (d + 0.0001) for d in distances]
    
w.transform = 'r'

formula = (
    '{outcome} ~ pct_frpm '
    '+ bach_pct + median_income + pct_el + diversity_idx '
    '+ locale + pct_swd + enroll_noncharter + dist_type + teaching_days + unemployment_pct + enroll_noncharter'
)

print("\n--- Moran's I on OLS Baseline (IDW k=8) ---")
for outcome in ['caaspp_ela', 'caaspp_math']:
    res = smf.ols(formula.format(outcome=outcome), data=df_spatial).fit()
    df_spatial[f'ols_resid_{outcome}'] = res.resid.values
    moran = Moran(res.resid.values, w)
    print(f"{outcome}: R² = {res.rsquared:.4f}, I = {moran.I:.4f}, p = {moran.p_sim:.4f}, z = {moran.z_norm:.4f}")

# endregion

# region FMA specs

MODEL_FEATURES = [
    'median_income',
    'bach_pct',
    'unemployment_pct',
    'poverty_pct',
    'locale',
    'pct_frpm',
    'pct_el',
    'pct_swd',
    'diversity_idx',
    'enroll_noncharter',
    'dist_type',
    'teaching_days',
]

MODEL_FEATURES = [
    c for c in MODEL_FEATURES
    if c in df_spatial.columns
]

# endregion

# region FMA random forest
from sklearn.compose import ColumnTransformer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

cv = KFold(n_splits=5, shuffle=True, random_state=42)

rf_base = RandomForestRegressor(
    n_estimators=1000,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

outcomes = ['caaspp_ela', 'caaspp_math']

rf_resids = {}
enet_resids = {}

print("\n" + "="*65)
print("NONLINEAR MODELS")
print("="*65)

for outcome in outcomes:

    print(f"\n── {outcome} ──")

    model_data = df_spatial[MODEL_FEATURES + [outcome, 'dist_name']].copy()
    model_data = model_data.dropna(subset=[outcome])

    cat_cols = ['locale', 'dist_type']
    cols_to_dummy = [c for c in cat_cols if c in model_data.columns]

    if cols_to_dummy:
        model_data = pd.get_dummies(
            model_data,
            columns=cols_to_dummy,
            dtype=float,
            drop_first=True
        )

    feature_cols = [
        c for c in model_data.columns
        if c not in [outcome, 'dist_name']
    ]

    X = model_data[feature_cols]
    y = model_data[outcome]

    num_cols = X.select_dtypes(include='number').columns

    preprocess = ColumnTransformer(
        [
            (
                'iterative',
                IterativeImputer(
                    estimator=BayesianRidge(),
                    random_state=42,
                    max_iter=10
                ),
                num_cols
            )
        ],
        remainder='passthrough'
    )

    enet_pipe = Pipeline([
        ('prep', preprocess),
        ('enet', ElasticNetCV(
            l1_ratio=[.1,.3,.5,.7,.9,.95,.99,1],
            cv=5,
            random_state=42,
            max_iter=10000
        ))
    ])

    enet_pred = cross_val_predict(
        enet_pipe,
        X,
        y,
        cv=cv
    )

    enet_r2 = r2_score(y, enet_pred)

    enet_resids[outcome] = pd.Series(
        y.values - enet_pred,
        index=model_data.index,
        name=f'enet_resid_{outcome}'
    )

    rf_pipe = Pipeline([
        ('prep', preprocess),
        ('rf', rf_base)
    ])

    rf_pred = cross_val_predict(
        rf_pipe,
        X,
        y,
        cv=cv
    )

    rf_r2 = r2_score(y, rf_pred)

    rf_resids[outcome] = pd.Series(
        y.values - rf_pred,
        index=model_data.index,
        name=f'rf_resid_{outcome}'
    )

    print(f"  Elastic Net CV R² : {enet_r2:.4f}")
    print(f"  Random Forest CV R²: {rf_r2:.4f}")

# endregion

# region composite PCA

combined = df_spatial[['dist_name', 'county']].copy()

for outcome in outcomes:

    combined[f'ols_resid_{outcome}'] = df_spatial[
        f'ols_resid_{outcome}'
    ].values

    combined = combined.join(
        enet_resids[outcome],
        how='left'
    )

    combined = combined.join(
        rf_resids[outcome],
        how='left'
    )

combined['enroll_noncharter_raw'] = np.exp(
    df_spatial.loc[df_spatial['dist_name'].isin(combined['dist_name']), 'enroll_noncharter'].values
)
min_enrollment = 30
combined = combined[combined['enroll_noncharter_raw'] >= min_enrollment].copy()
combined = combined.merge(df_spatial[['dist_name', 'lat', 'lon']], on='dist_name', how='left')
combined = combined.merge(df_spatial[['dist_name', 'locale']], on='dist_name', how='left')

from scipy.stats import zscore
combined['ols_ela_z'] = zscore(
    combined['ols_resid_caaspp_ela'],
    nan_policy='omit'
)

combined['ols_math_z'] = zscore(
    combined['ols_resid_caaspp_math'],
    nan_policy='omit'
)

combined['enet_ela_z'] = zscore(
    combined['enet_resid_caaspp_ela'],
    nan_policy='omit'
)

combined['enet_math_z'] = zscore(
    combined['enet_resid_caaspp_math'],
    nan_policy='omit'
)

combined['rf_ela_z'] = zscore(
    combined['rf_resid_caaspp_ela'],
    nan_policy='omit'
)

combined['rf_math_z'] = zscore(
    combined['rf_resid_caaspp_math'],
    nan_policy='omit'
)

components = [
    'ols_ela_z',
    'ols_math_z',
    'enet_ela_z',
    'enet_math_z',
    'rf_ela_z',
    'rf_math_z'
]

for col in components:
    lower_bound = combined[col].quantile(0.01)
    upper_bound = combined[col].quantile(0.99)
    combined[col] = combined[col].clip(lower=lower_bound, upper=upper_bound)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca_raw = pca.fit_transform(combined[components])

combined['pca_raw'] = pca_raw
if combined[components[0]].corr(combined['pca_raw']) < 0:
    combined['pca_raw'] *= -1

print(f"  -> PCA loadings: {dict(zip(components, pca.components_[0].round(3)))}")
print(f"  -> Variance explained by PC1: {pca.explained_variance_ratio_[0]*100:.1f}%")

# endregion

# region bootstrap PCA

from sklearn.decomposition import PCA
from sklearn.utils import resample
import numpy as np

B = 1000          # 500 is acceptable; 1000 is preferable
rng = 42

X = combined[components].values
n = len(combined)

boot_scores = np.empty((B, n))

# Original PCA direction for sign alignment
pca_ref = PCA(n_components=1)
pca_ref.fit(X)
ref_loading = pca_ref.components_[0]

for b in range(B):

    # Bootstrap districts
    boot_idx = resample(
        np.arange(n),
        replace=True,
        n_samples=n,
        random_state=rng + b
    )

    X_boot = X[boot_idx]

    pca_boot = PCA(n_components=1)
    pca_boot.fit(X_boot)

    # Keep PC direction consistent
    if np.dot(pca_boot.components_[0], ref_loading) < 0:
        pca_boot.components_[0] *= -1

    # Project ALL districts onto this bootstrap PC
    boot_scores[b] = pca_boot.transform(X).ravel()

combined["pc1_var"] = boot_scores.var(axis=0, ddof=1)
combined["pc1_se"]  = np.sqrt(combined["pc1_var"])

print("\nBootstrap PCA complete")
print(f"Median PC1 SE : {combined['pc1_se'].median():.4f}")
print(f"Mean   PC1 SE : {combined['pc1_se'].mean():.4f}")
print(f"Max    PC1 SE : {combined['pc1_se'].max():.4f}")

# endregion

# region empirical bayes
import numpy as np
import pandas as pd
import spreg
from libpysal.weights.spatial_lag import lag_spatial
from scipy.spatial.distance import euclidean
from scipy.stats import zscore
from libpysal.weights import KNN

score_col = 'pca_raw'
group_col = 'locale'
n_col = 'enroll_noncharter_raw'

eb_score = pd.Series(index=combined.index, dtype=float)
combined['v_i'] = np.nan
combined['tau_sq'] = np.nan
combined['posterior_var'] = np.nan

print("\n" + "="*50)
print("EMPIRICAL BAYES SHRINKAGE (GLOBAL POPULATION WEIGHT)")
print("="*50)


global_median_n = combined[n_col].median()
global_large = combined.loc[combined[n_col] >= global_median_n, score_col]
global_small = combined.loc[combined[n_col] < global_median_n, score_col]

global_tau_sq = np.var(global_large, ddof=1)
global_small_var = np.var(global_small, ddof=1)
global_noise_var = max(0.0, global_small_var - global_tau_sq)

global_mean_inv_n = np.mean(1.0 / combined.loc[combined[n_col] < global_median_n, n_col])
global_sigma_sq = global_noise_var / global_mean_inv_n if global_mean_inv_n > 0 else 0.0

print(f"Global Student Noise Parameter (sigma_sq): {global_sigma_sq:.3f}\n")


for name, group in combined.groupby(group_col):
    y = group[score_col].values
    n = group[n_col].values
    pc1_var = group['pc1_var'].values
    
    y_bar = np.average(y, weights=n)
    
    if len(y) < 3:
        eb_score.loc[group.index] = y
        combined.loc[group.index, 'v_i'] = pc1_var
        combined.loc[group.index, 'posterior_var'] = pc1_var
        continue
        
    median_n_loc = np.median(n)
    large_districts = y[n >= median_n_loc]
    tau_sq = np.var(large_districts, ddof=1) if len(large_districts) > 1 else np.var(y, ddof=1)
    tau_sq = max(0.01, tau_sq)
    
    v_i = (global_sigma_sq / n) + pc1_var
    
    w_i = tau_sq / (tau_sq + v_i)
    
    # Calculate POSTERIOR VARIANCE for the Monte Carlo Simulation
    posterior_var = w_i * v_i
    
    eb_score.loc[group.index] = y_bar + w_i * (y - y_bar)
    
    combined.loc[group.index, 'v_i'] = v_i
    combined.loc[group.index, 'tau_sq'] = tau_sq
    combined.loc[group.index, 'posterior_var'] = posterior_var
    
    print(f"  Locale: {name:<10} | Local Signal (tau_sq): {tau_sq:.3f}")

combined['eb_shrunken_raw'] = eb_score

# Spatial smoothing
coords_sub = combined[['lon', 'lat']].values
w_sub = KNN.from_array(coords_sub, k=8)

for i, neighbors in w_sub.neighbors.items():
    distances = [euclidean(coords_sub[i], coords_sub[j]) for j in neighbors]
    w_sub.weights[i] = [1.0 / (d + 0.0001) for d in distances]
w_sub.transform = 'r'

X = np.ones((len(combined), 1))
y_sar = combined['eb_shrunken_raw'].values.reshape(-1, 1)

sar_model = spreg.ML_Lag(y_sar, X, w=w_sub, name_y='eb_shrunken_raw')
rho = max(0.0, min(sar_model.rho, 0.5))
neighbor_avg = lag_spatial(w_sub, combined['eb_shrunken_raw'].values)

combined['sar_smoothed'] = (1 - rho) * combined['eb_shrunken_raw'] + rho * neighbor_avg
combined['pca_z'] = zscore(combined['sar_smoothed'], nan_policy='omit')

final_lower = combined['pca_z'].quantile(0.01)
final_upper = combined['pca_z'].quantile(0.99)
combined['pca_z'] = combined['pca_z'].clip(lower=final_lower, upper=final_upper)

# endregion

# region elastic net

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

enet_resids = {}

print("\n" + "=" * 65)
print("ELASTIC NET")
print("=" * 65)

for outcome in ['caaspp_ela', 'caaspp_math']:

    print(f"\n── {outcome} ──")

    model_data = df_spatial[
        MODEL_FEATURES + [outcome, 'dist_name']
    ].copy()

    model_data = model_data.dropna(subset=[outcome])

    cat_cols = [
        c for c in ['locale', 'dist_type']
        if c in model_data.columns
    ]

    if cat_cols:
        model_data = pd.get_dummies(
            model_data,
            columns=cat_cols,
            dtype=float,
            drop_first=True
        )

    feature_cols = [
        c for c in model_data.columns
        if c not in [outcome, 'dist_name']
    ]

    X = model_data[feature_cols]
    y = model_data[outcome]

    num_cols = X.select_dtypes(include='number').columns

    preprocess = ColumnTransformer(
        [
            (
                'imputer',
                IterativeImputer(
                    estimator=BayesianRidge(),
                    random_state=42,
                    max_iter=10
                ),
                num_cols
            )
        ],
        remainder='passthrough'
    )

    enet = Pipeline([
        ('prep', preprocess),
        ('enet', ElasticNetCV(
            l1_ratio=[0.05, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
            cv=5,
            random_state=42,
            max_iter=10000
        ))
    ])

    pred = cross_val_predict(
        enet,
        X,
        y,
        cv=cv,
        n_jobs=-1
    )

    print(f"  CV R²: {r2_score(y, pred):.4f}")

    enet_resids[outcome] = pd.Series(
        y.values - pred,
        index=model_data.index,
        name=f'enet_resid_{outcome}'
    )

# endregion

# region validation

import seaborn as sns

print("\n" + "="*50)
print("CHECK 1: TAIL SHRINKAGE ANALYSIS")
print("="*50)

# Calculate raw OLS z-scores for comparison
combined['ols_ela_z'] = zscore(combined['ols_resid_caaspp_ela'], nan_policy='omit')
combined['ols_math_z'] = zscore(combined['ols_resid_caaspp_math'], nan_policy='omit')
combined['avg_ols_z'] = (combined['ols_ela_z'] + combined['ols_math_z']) / 2

fig, ax = plt.subplots(figsize=(10, 8))

# Scatter raw OLS vs Final Optimized PCA
sns.scatterplot(x=combined['avg_ols_z'], y=combined['pca_z'], alpha=0.6, ax=ax)

# Add a 1:1 reference line
limits = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  
    np.max([ax.get_xlim(), ax.get_ylim()]),  
]
ax.plot(limits, limits, 'r--', label='1:1 Line (No Shrinkage Effect)')

ax.set_title("Effect of Shrinkage: Raw OLS vs. Optimized PCA Z-Score")
ax.set_xlabel("Raw OLS Average Z-Score (Unshrunken)")
ax.set_ylabel("Final PCA Z-Score (Optimized Latent Factor)")
ax.legend()
plt.tight_layout()
plt.savefig("tail_shrinkage_check.png")
plt.show()

# Examine the variance ratio
variance_retained = combined['pca_z'].var() / combined['avg_ols_z'].var()
print(f"Variance retained in final PCA vs raw OLS: {variance_retained:.2%}")
if variance_retained < 0.5:
    print("WARNING: You have lost more than half of your variance. Extreme outliers may be erased.")


print("\n" + "="*50)
print("CHECK 2: RANK STABILITY (SPEARMAN CORRELATION)")
print("="*50)

import scipy.stats as stats


spearman_corr, p_value = stats.spearmanr(
    combined['pca_z'], 
    combined['avg_ols_z'], 
    nan_policy='omit'
)

print(f"Spearman Rank Correlation (Final PCA vs. Raw OLS): {spearman_corr:.4f}")
print(f"p-value: {p_value:.4e}")

if spearman_corr > 0.95:
    print("  -> Result: EXTREMELY HIGH agreement. Shrinkage adjusted magnitudes, but general rank order is largely unchanged.")
elif spearman_corr > 0.80:
    print("  -> Result: HIGH agreement. Ranks are stable, with meaningful targeted reordering by Empirical Bayes/RF.")
elif spearman_corr > 0.50:
    print("  -> Result: MODERATE agreement. The ensembling pipeline substantially reshuffled the baseline district rankings.")
else:
    print("  -> Result: LOW agreement. The final model fundamentally transformed the baseline OLS rankings.")

print("\n" + "="*50)
print("CHECK 3: SUBGROUP BIAS AND VARIANCE CHECK")
print("="*50)

check_df = combined.merge(
    df_spatial[['dist_name', 'enroll_noncharter', 'dist_type', 'pct_frpm']], 
    on='dist_name', 
    how='left'
)

check_df['enrollment_raw'] = np.exp(check_df['enroll_noncharter'])

check_df['size_quintile'] = pd.qcut(
    check_df['enrollment_raw'], 
    q=5, 
    labels=['Smallest', 'Small', 'Medium', 'Large', 'Largest']
)

print("\n--- Mean PCA Z-Score by District Size ---")
print(check_df.groupby('size_quintile', observed=False)['pca_z'].mean().round(3))

print("\n--- Mean PCA Z-Score by Locale ---")
print(check_df.groupby('locale', observed=False)['pca_z'].mean().round(3))

print("\n--- Variance of PCA Z-Score by District Size ---")
print(check_df.groupby('size_quintile', observed=False)['pca_z'].var().round(3))

corr_frpm = check_df['pca_z'].corr(check_df['pct_frpm'])
print(f"\nCorrelation between Final PCA Score and Free/Reduced Lunch %: {corr_frpm:.3f}")
if abs(corr_frpm) > 0.3:
    print("WARNING: Residuals are still heavily correlated with poverty. Model is missing structural covariates.")

# endregion

# region STABILITY

# region STABILITY 1: INTERNAL COMPONENT AGREEMENT
print("\n" + "="*50)
print("STABILITY 1: INTERNAL MODEL AGREEMENT (SUBJECT SPREAD)")
print("="*50)

rank_cols = []
for col in components:
    rank_col = f'rank_{col}'
    combined[rank_col] = combined[col].rank(ascending=False, method='min')
    rank_cols.append(rank_col)

combined['component_rank_std'] = combined[rank_cols].std(axis=1)
combined['component_rank_spread'] = combined[rank_cols].max(axis=1) - combined[rank_cols].min(axis=1)

print("\n--- Most Stable Districts (Top 10 by Lowest Rank Spread) ---")
print(combined.sort_values('component_rank_spread').head(10)[['dist_name', 'pca_z', 'component_rank_spread']])

print("\n--- Most Volatile Districts (Top 10 by Highest Rank Spread) ---")
print(combined.sort_values('component_rank_spread', ascending=False).head(10)[['dist_name', 'pca_z', 'component_rank_spread']])

# endregion

# region STABILITY 2: MONTE CARLO RANK SIMULATION


print("\n" + "="*50)
print("STABILITY 2: MONTE CARLO RANK SIMULATION (SCALE-CORRECTED)")
print("="*50)

ela_cols = ['ols_ela_z', 'enet_ela_z', 'rf_ela_z']
math_cols = ['ols_math_z', 'enet_math_z', 'rf_math_z']


sd_sar = combined['sar_smoothed'].std(ddof=1)

combined['posterior_var_z'] = (combined['posterior_var'] * (1 - rho)**2) / (sd_sar**2)

combined['total_variance_z'] = combined['posterior_var_z']
combined['se_z'] = np.sqrt(combined['total_variance_z'])

print(f"  -> Median RAW posterior variance:          {combined['posterior_var'].median():.5f}")
print(f"  -> Median SCALED posterior variance:       {combined['posterior_var_z'].median():.5f}")
print(f"  -> Median total standard error (se_z):     {combined['se_z'].median():.5f}")



n_sims = 1000000
n_districts = len(combined)
simulated_ranks = np.zeros((n_districts, n_sims))

noise = np.random.normal(loc=0, scale=combined['se_z'].values[:, None], size=(n_districts, n_sims))
simulated_scores = combined['pca_z'].values[:, None] + noise

for i in range(n_sims):
    simulated_ranks[:, i] = pd.Series(simulated_scores[:, i]).rank(ascending=False, method='min')

combined['rank_final'] = combined['pca_z'].rank(ascending=False, method='min')
combined['rank_95_best'] = np.percentile(simulated_ranks, 2.5, axis=1)
combined['rank_95_worst'] = np.percentile(simulated_ranks, 97.5, axis=1)
combined['plus_minus_spots'] = (combined['rank_95_worst'] - combined['rank_95_best']) / 2

print("\n--- Top 5 Overall Districts ---")
top_5 = combined.sort_values('rank_final').head(5)
for _, row in top_5.iterrows():
    print(f"{row['dist_name'][:30]:<30} | Rank: {int(row['rank_final'])} (+/- {int(row['plus_minus_spots'])} spots) | Range: {int(row['rank_95_best'])} to {int(row['rank_95_worst'])}")

print("\n--- Median Districts ---")
sorted_combined = combined.sort_values('rank_final')
mid_index = len(sorted_combined) // 2

median_5 = sorted_combined.iloc[mid_index - 2 : mid_index + 3]
for _, row in median_5.iterrows():
    print(f"{row['dist_name'][:30]:<30} | Rank: {int(row['rank_final'])} (+/- {int(row['plus_minus_spots'])} spots) | Range: {int(row['rank_95_best'])} to {int(row['rank_95_worst'])}")

print("\n--- Bottom 5 Overall Districts ---")
bottom_5 = combined.sort_values('rank_final', ascending=False).head(5)
for _, row in bottom_5.iterrows():
    print(f"{row['dist_name'][:30]:<30} | Rank: {int(row['rank_final'])} (+/- {int(row['plus_minus_spots'])} spots) | Range: {int(row['rank_95_best'])} to {int(row['rank_95_worst'])}")

# endregion

# endregion

# region TIER ASSIGNMENT

print("\n" + "="*50)
print("TIER ASSIGNMENT: CATEGORIZING PERFORMANCE")
print("="*50)

if 'assist_status' in combined.columns:
    combined = combined.drop(columns=['assist_status'])

combined = combined.merge(df_spatial[['dist_name', 'assist_status']], on='dist_name', how='left')

combined['final_z_standardized'] = combined['pca_z']

def assign_tier(z, spread): 
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

combined['performance_tier'] = combined.apply(
    lambda row: assign_tier(row['final_z_standardized'], row['component_rank_spread']), 
    axis=1
)

print("\n--- Distribution of Districts by Tier ---")
print(combined['performance_tier'].value_counts())
print("-" * 30)

output_cols = [
    'dist_name', 'performance_tier', 'final_z_standardized', 
    'plus_minus_spots', 'component_rank_spread', 'assist_status'
]

export_df = combined[output_cols].sort_values('final_z_standardized', ascending=False)
export_df.to_csv("district_performance_tiers.csv", index=False)
print("\nExported final tiered rankings to 'district_performance_tiers.csv'")

# endregion

# endregion




# region VALIDATION: EXTERNAL CDS AWARDS (bayes smoothed)

import numpy as np

print("\n" + "="*50)
print("EXTERNAL VALIDATION: CA DISTINGUISHED SCHOOLS (BAYESIAN SMOOTHED)")
print("="*50)

try:
    # 1. Load the award file
    try:
        awards = pd.read_csv("data/2024-25award.csv")
    except FileNotFoundError:
        awards = pd.read_csv("data/dsaawards.xlsx - CA Distinguished Schools.csv")
    
    recent_awards = awards[awards['Year'].isin([2024, 2025])].copy()
    if recent_awards.empty:
        recent_awards = awards[awards['Year'] == awards['Year'].max()].copy()

    # 2. Robust CDS Code Merging (Fallback to strings if missing)
    school_counts = final[['dist_name', 'cds_code', 'agency_name_clean', 'n_schools']].copy()
    
    award_cds_col = next((col for col in recent_awards.columns if 'cds' in col.lower()), None)
    
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
    
    # --- BAYESIAN SMOOTHING PARAMETER ESTIMATION ---
    raw_rates = val_df['award_count'] / val_df['n_schools']
    global_mean = raw_rates.mean()
    
    # Use weighted variance to prevent single-school districts from destroying priors
    weighted_var = np.average((raw_rates - global_mean)**2, weights=val_df['n_schools'])
    
    if weighted_var > 0 and weighted_var < (global_mean * (1 - global_mean)):
        gamma = (global_mean * (1 - global_mean) / weighted_var) - 1
        alpha = global_mean * gamma
        beta = (1 - global_mean) * gamma
    else:
        # Robust fallback: Set prior weight to median schools per district
        prior_weight = val_df['n_schools'].median()
        alpha = global_mean * prior_weight
        beta = (1 - global_mean) * prior_weight

    print(f"  -> Empirical Bayes Priors: alpha = {alpha:.3f}, beta = {beta:.3f}")
    
    val_df['cds_density_smoothed'] = (val_df['award_count'] + alpha) / (val_df['n_schools'] + alpha + beta)
    
    val_df_final = combined.merge(val_df[['dist_name', 'cds_density_smoothed', 'award_count']], on='dist_name', how='left')
    val_df_final['cds_density_smoothed'] = val_df_final['cds_density_smoothed'].fillna(alpha / (alpha + beta))
    val_df_final['award_count'] = val_df_final['award_count'].fillna(0)

    # 5. Robust Rank Correlation handling massive blocks of tied zeroes
    if val_df_final['cds_density_smoothed'].std() == 0:
        print("  -> ERROR: Smoothed density has zero variance.")
    else:
        # Swap Spearman for Kendall's Tau-b
        tau, p_val = stats.kendalltau(
            val_df_final['final_z_standardized'], 
            val_df_final['cds_density_smoothed'], 
            nan_policy='omit'
        )
        
        print(f"  -> Diagnostic: Mapped {int(val_df['award_count'].sum())} awards across active dataset.")
        print(f"  -> Kendall's Tau-b (tie-adjusted): {tau:.4f}")
        print(f"  -> p-value: {p_val:.4e}")
        
        if tau > 0.20:
            print("  -> Result: STRONG validation.")
        elif tau > 0.10:
            print("  -> Result: MODERATE validation.")
        elif tau > 0:
            print("  -> Result: WEAK validation.")
        else:
            print("  -> Result: INVERSE/NO validation.")

except Exception as e:
    print(f"  -> WARNING: Could not complete validation. Error: {e}")

# endregion

# region tables 

# region demographic table: overperformers vs underperformers

demo = final.copy()

tier_df = pd.read_csv("district_performance_tiers.csv")

demo = demo.merge(
    tier_df[['dist_name', 'performance_tier']],
    on='dist_name',
    how='inner'
)

overperformers = demo[
    demo['performance_tier'].isin([
        'Moderate Overperformer',
        'Significant Overperformer'
    ])
]

underperformers = demo[
    demo['performance_tier'].isin([
        'Moderate Underperformer',
        'Significant Underperformer'
    ])
]

def summarize(group):
    return {
        'Rural Districts (%)':
            100 * group['locale'].eq('Rural').mean(),
        'Average Enrollment':
            group['enrollment'].mean(),
        'Ethnic Diversity Index':
            group['diversity_idx'].mean(),
        'English Learners (%)':
            group['pct_el'].mean(),
        'Free/Reduced Meals (%)':
            group['pct_frpm'].mean(),
        "Adults with Bachelor's Degree (%)":
            group['bach_pct'].mean()
    }

over_stats = summarize(overperformers)
under_stats = summarize(underperformers)

table = pd.DataFrame({
    f'Overperformers (n={len(overperformers)})': over_stats,
    f'Underperformers (n={len(underperformers)})': under_stats
})

table.loc['Average Enrollment'] = table.loc['Average Enrollment'].round(0)
table = table.round(1)

print("\nDEMOGRAPHIC COMPARISON")
print("=" * 60)
print(table)

# endregion

# region demographic table: significant overperformers vs significant underperformers

sig_overperformers = demo[
    demo['performance_tier'] == 'Significant Overperformer'
]

sig_underperformers = demo[
    demo['performance_tier'] == 'Significant Underperformer'
]

sig_over_stats = summarize(sig_overperformers)
sig_under_stats = summarize(sig_underperformers)

sig_table = pd.DataFrame({
    f'Significant Overperformers (n={len(sig_overperformers)})': sig_over_stats,
    f'Significant Underperformers (n={len(sig_underperformers)})': sig_under_stats
})

sig_table.loc['Average Enrollment'] = (
    sig_table.loc['Average Enrollment'].round(0)
)

sig_table = sig_table.round(1)

print("\nSIGNIFICANT-TIER DEMOGRAPHIC COMPARISON")
print("=" * 60)
print(sig_table)

# endregion

# region district assistance status by performance tier

assistance_df = final.copy()

tier_df = pd.read_csv("district_performance_tiers.csv")

assistance_df = assistance_df.merge(
    tier_df[["dist_name", "performance_tier"]],
    on="dist_name",
    how="inner"
)

assistance_df["assistance_type"] = np.where(
    assistance_df["assist_status"].str.contains(
        "Differentiated", case=False, na=False
    ),
    "Differentiated",
    "General"
)

tier_order = [
    "Moderate Underperformer",
    "Expected Performer",
    "Moderate Overperformer",
    "Significant Underperformer",
    "Significant Overperformer"
]

assistance_table = (
    pd.crosstab(
        assistance_df["performance_tier"],
        assistance_df["assistance_type"]
    )
    .reindex(tier_order)
    .fillna(0)
    .astype(int)
)

assistance_table["Total"] = assistance_table.sum(axis=1)
assistance_table["% Differentiated"] = (
    assistance_table["Differentiated"]
    / assistance_table["Total"]
    * 100
)

assistance_table = assistance_table[
    ["Differentiated", "% Differentiated", "General", "Total"]
]

total_diff = assistance_table["Differentiated"].sum()
total_gen = assistance_table["General"].sum()
total_n = assistance_table["Total"].sum()

assistance_table.loc["Total"] = [
    total_diff,
    100 * total_diff / total_n,
    total_gen,
    total_n
]

display_table = assistance_table.copy()
display_table["% Differentiated"] = (
    display_table["% Differentiated"]
    .round(1)
    .astype(str)
    + "%"
)

print("\n--- District Assistance Status by Performance Tier ---")
print(display_table)

# endregion

# region baseline characteristics

table1_df = final.merge(df_spatial[['dist_name']], on='dist_name', how='inner')

def fmt_continuous(group, col):
    vals = group[col].dropna()
    if len(vals) == 0:
        return "—"
    return f"{vals.mean():.1f} ({vals.std():.1f})"

def fmt_categorical(group, col, level):
    vals = group[col].dropna()
    if len(vals) == 0:
        return "0.0%"
    return f"{100 * vals.eq(level).mean():.1f}%"

def baseline_summarize(group):
    return {
        # CAASPP Outcomes
        "CAASPP ELA Score, mean (SD)": 
            fmt_continuous(group, "caaspp_ela"),
        "CAASPP Math Score, mean (SD)": 
            fmt_continuous(group, "caaspp_math"),
        # Model Features
        "Median Household Income, mean (SD)": 
            fmt_continuous(group, "median_income"),
        "Adults w/ Bachelor's Degree, % mean (SD)": 
            fmt_continuous(group, "bach_pct"),
        "Unemployment, % mean (SD)": 
            fmt_continuous(group, "unemployment_pct"),
        "Poverty, % mean (SD)": 
            fmt_continuous(group, "poverty_pct"),
        "Free/Reduced Meals, % mean (SD)": 
            fmt_continuous(group, "pct_frpm"),
        "English Learners, % mean (SD)": 
            fmt_continuous(group, "pct_el"),
        "Students w/ Disabilities, % mean (SD)": 
            fmt_continuous(group, "pct_swd"),
        "Ethnic Diversity Index, mean (SD)": 
            fmt_continuous(group, "diversity_idx"),
        "Log Enrollment (Non-Charter), mean (SD)": 
            fmt_continuous(group, "enroll_noncharter"),
        "Teaching Days, mean (SD)": 
            fmt_continuous(group, "teaching_days"),
        "District Type: Elementary, %": 
            fmt_categorical(group, "dist_type", "Elementary"),
        "District Type: High School, %": 
            fmt_categorical(group, "dist_type", "High School"),
        "District Type: Unified, %": 
            fmt_categorical(group, "dist_type", "Unified"),
    }

locales = ['City', 'Suburb', 'Town', 'Rural']

baseline_columns = {}
for loc in locales:
    grp = table1_df[table1_df["locale"] == loc]
    baseline_columns[f"{loc} (n={len(grp)})"] = baseline_summarize(grp)

baseline_columns[f"Total (n={len(table1_df)})"] = baseline_summarize(table1_df)

baseline_table = pd.DataFrame(baseline_columns)

print("\nTABLE 1. BASELINE CHARACTERISTICS BY LOCALITY")
print("=" * 70)
print(baseline_table)

# endregion

# endregion 

# region PLOTS

# region CERF CA Bar Chart 

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig, ax = plt.subplots(figsize=(13, 11.5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

colors = ['#8b0000', '#d85b5b', '#bdbdbd', '#80cbc4', '#004d40']

cerf_region_map = {
    'Imperial': 'Southern Border', 'San Diego': 'Southern Border',
    'Orange': 'Orange County',
    'Los Angeles': 'Los Angeles County',
    'Riverside': 'Inland Empire', 'San Bernardino': 'Inland Empire',
    'Monterey': 'Central Coast', 'San Benito': 'Central Coast', 'Santa Barbara': 'Central Coast', 'Santa Cruz': 'Central Coast', 'San Luis Obispo': 'Central Coast', 'Ventura': 'Central Coast',
    'Fresno': 'Central San Joaquin Valley', 'Kings': 'Central San Joaquin Valley', 'Madera': 'Central San Joaquin Valley', 'Tulare': 'Central San Joaquin Valley',
    'Kern': 'Kern County',
    'Merced': 'Northern San Joaquin Valley', 'San Joaquin': 'Northern San Joaquin Valley', 'Stanislaus': 'Northern San Joaquin Valley',
    'Alpine': 'Eastern Sierra', 'Amador': 'Eastern Sierra', 'Calaveras': 'Eastern Sierra', 'Inyo': 'Eastern Sierra', 'Mariposa': 'Eastern Sierra', 'Mono': 'Eastern Sierra', 'Tuolumne': 'Eastern Sierra',
    'Alameda': 'Bay Area', 'Contra Costa': 'Bay Area', 'Marin': 'Bay Area', 'Napa': 'Bay Area', 'San Francisco': 'Bay Area', 'San Mateo': 'Bay Area', 'Santa Clara': 'Bay Area', 'Solano': 'Bay Area', 'Sonoma': 'Bay Area',
    'Colusa': 'Sacramento', 'El Dorado': 'Sacramento', 'Nevada': 'Sacramento', 'Placer': 'Sacramento', 'Sacramento': 'Sacramento', 'Sutter': 'Sacramento', 'Yolo': 'Sacramento', 'Yuba': 'Sacramento',
    'Del Norte': 'Redwood Coast', 'Humboldt': 'Redwood Coast', 'Lake': 'Redwood Coast', 'Mendocino': 'Redwood Coast',
    'Butte': 'North State', 'Glenn': 'North State', 'Lassen': 'North State', 'Modoc': 'North State', 'Plumas': 'North State', 'Shasta': 'North State', 'Sierra': 'North State', 'Siskiyou': 'North State', 'Tehama': 'North State', 'Trinity': 'North State'
}

combined['cerf_region'] = combined['county'].map(cerf_region_map).fillna('Unclassified')

bins = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
labels = ["Sig. Under", "Mod. Under", "Expected", "Mod. Over", "Sig. Over"]
combined['tier'] = pd.cut(combined['pca_z'], bins=bins, labels=labels)

ct = pd.crosstab(combined['cerf_region'], combined['tier'], normalize='index') * 100

ct = pd.crosstab(combined['cerf_region'], combined['tier'], normalize='index') * 100
ct = ct.reindex(columns=labels, fill_value=0)

n_counts = combined.groupby('cerf_region').size()
ct.index = [f"{reg} (n={n_counts[reg]})" if reg in n_counts else reg for reg in ct.index]

ct['delta'] = (ct['Mod. Over'] + ct['Sig. Over']) - (ct['Sig. Under'] + ct['Mod. Under'])
ct = ct.sort_values('delta', ascending=True).drop(columns='delta')

is_unclassified = ct.index.str.contains('Unclassified', case=False, na=False)
valid_regions = ct[~is_unclassified]
unclassified_region = ct[is_unclassified]
ct = pd.concat([unclassified_region, valid_regions])


half_exp = ct['Expected'] / 2

mod_under_right = -half_exp
mod_under_left = mod_under_right - ct['Mod. Under']
sig_under_right = mod_under_left
sig_under_left = sig_under_right - ct['Sig. Under']

mod_over_left = half_exp
mod_over_right = mod_over_left + ct['Mod. Over']
sig_over_left = mod_over_right
sig_over_right = sig_over_left + ct['Sig. Over']

ax.barh(ct.index, half_exp, left=mod_under_right, color=colors[2], label='At Baseline (Expected)', height=0.65, edgecolor='white', linewidth=0.6)
ax.barh(ct.index, half_exp, left=0, color=colors[2], height=0.65, edgecolor='white', linewidth=0.6)
ax.barh(ct.index, -ct['Mod. Under'], left=mod_under_right, color=colors[1], label='Moderate Underperformer', height=0.65, edgecolor='white', linewidth=0.6)
ax.barh(ct.index, -ct['Sig. Under'], left=sig_under_right, color=colors[0], label='Significant Underperformer', height=0.65, edgecolor='white', linewidth=0.6)
ax.barh(ct.index, ct['Mod. Over'], left=mod_over_left, color=colors[3], label='Moderate Overperformer', height=0.65, edgecolor='white', linewidth=0.6)
ax.barh(ct.index, ct['Sig. Over'], left=sig_over_left, color=colors[4], label='Significant Overperformer', height=0.65, edgecolor='white', linewidth=0.6)

import matplotlib.patheffects as pe

label_outline = [pe.withStroke(linewidth=2.5, foreground='#1b1b1b')]

ZERO_TOL = 0.05


segments = [
    ('Mod. Under', mod_under_left, mod_under_right),
    ('Sig. Under', sig_under_left, sig_under_right),
    ('Mod. Over', mod_over_left, mod_over_right),
    ('Sig. Over', sig_over_left, sig_over_right),
]

def _edge_at(edge, region):
    return edge[region] if isinstance(edge, pd.Series) else edge

for col, left_edges, right_edges in segments:
    for region in ct.index:
        w = ct.loc[region, col]
        if w < ZERO_TOL:
            continue
        left = _edge_at(left_edges, region)
        right = _edge_at(right_edges, region)
        center_x = (left + right) / 2
        # Shrink font slightly for very narrow segments so the outlined text
        # has a better chance of fitting within (or just over) its sliver.
        fontsize = 9 if w >= 4 else 7.5
        ax.text(
            center_x, region, f"{w:.0f}%",
            ha='center', va='center',
            fontsize=fontsize, color='white',
            path_effects=label_outline,
            zorder=6
        )
for region in ct.index:
    w = ct.loc[region, 'Expected']
    if w < ZERO_TOL:
        continue
    fontsize = 9 if w >= 4 else 7.5
    ax.text(
        0, region, f"{w:.0f}%",
        ha='center', va='center',
        fontsize=fontsize, color='#1b1b1b',
        path_effects=[pe.withStroke(linewidth=2.5, foreground='white')],
        zorder=6
    )

ax.axvline(0, color='#1b1b1b', linewidth=1.5, zorder=5)

fig.suptitle("Regional Performance Gaps", fontsize=22, weight='bold', color='#1b1b1b', y=0.96, x=0.5, ha='center')

ax.spines[['top', 'right', 'left']].set_visible(False)
ax.spines['bottom'].set_color('#cccccc')
ax.spines['bottom'].set_linewidth(1.0)

left_extent = sig_under_left.min()
right_extent = sig_over_right.max()
margin = 5
ax.set_xlim(left_extent - margin, right_extent + margin)

ax.tick_params(axis='x', labelsize=10, colors='#1b1b1b')
ax.tick_params(axis='y', labelsize=11, colors='#1b1b1b', length=0, pad=12)
xticks = ax.get_xticks()
ax.set_xticklabels([f"{abs(int(x))}%" for x in xticks])

ax.set_xlabel("Percentage of Districts Deviating from Baseline", fontsize=11, labelpad=15, color='#1b1b1b')

ax.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.06), 
    ncol=5, 
    frameon=False,
    fontsize=10,
    columnspacing=1.5
)

fig.text(
    0.05, 0.01, 
    "*Note: Regions reflect official CERF CA economic region designations.", 
    fontsize=9, style='italic', color='#5b5b5b', ha='left', va='bottom'
)

plt.subplots_adjust(top=0.90, bottom=0.12, left=0.20)
plt.savefig("cerf_regional_gaps_final.png", dpi=300, bbox_inches='tight')
plt.show()

# endregion

# %%
# %%
# %%
# region clusters (SES vs. Performance)

import matplotlib.transforms as transforms
import re
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from adjustText import adjust_text
import matplotlib.pyplot as plt

print("\n" + "="*60)
print("VISUALIZATION: SES VS. PERFORMANCE DIAGNOSTIC")
print("="*60)

ses_cols = ['pct_frpm', 'median_income', 'bach_pct', 'unemployment_pct']
cluster_df = combined.merge(df_spatial[['dist_name'] + ses_cols], on='dist_name', how='left')

scaler_ses = StandardScaler()
ses_scaled = scaler_ses.fit_transform(cluster_df[ses_cols])
pca_ses = PCA(n_components=1)
cluster_df['ses_index'] = pca_ses.fit_transform(ses_scaled)

if cluster_df['ses_index'].corr(cluster_df['median_income']) < 0:
    cluster_df['ses_index'] *= -1

def assign_strategic_group(row):
    if row['ses_index'] >= 0 and row['pca_z'] >= 0:
        return "Expected High"
    elif row['ses_index'] < 0 and row['pca_z'] >= 0:
        return "Beat-the-Odds"
    elif row['ses_index'] < 0 and row['pca_z'] < 0:
        return "Systemic Challenge"
    else:
        return "Underutilized"

cluster_df['strategic_group'] = cluster_df.apply(assign_strategic_group, axis=1)

cluster_df['dist_name_short'] = cluster_df['dist_name'].apply(
    lambda s: re.sub(r'\s*\([^)]*\)\s*$', '', s)
)

COLOR_OVERPERFORM  = "#14532D"
COLOR_UNDERPERFORM = "#7A271A"
COLOR_BG           = "#C9CCCF"
COLOR_TEXT         = "#1B1B1B"
COLOR_GRID         = "#DFE1E2"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["text.color"] = COLOR_TEXT
plt.rcParams["axes.edgecolor"] = COLOR_TEXT
plt.rcParams["axes.labelcolor"] = COLOR_TEXT
plt.rcParams["xtick.color"] = COLOR_TEXT
plt.rcParams["ytick.color"] = COLOR_TEXT

N_LABELS = 20  # number of over- and under-performers shown

top_over  = cluster_df[cluster_df['strategic_group'] == "Beat-the-Odds"].sort_values('pca_z', ascending=False).head(N_LABELS)
top_under = cluster_df[cluster_df['strategic_group'] == "Underutilized"].sort_values('pca_z', ascending=True).head(N_LABELS)

lollipop_df = pd.concat([top_under, top_over], ignore_index=True)
lollipop_df = lollipop_df.sort_values('pca_z', ascending=True).reset_index(drop=True)

bar_colors = [COLOR_OVERPERFORM if v >= 0 else COLOR_UNDERPERFORM for v in lollipop_df['pca_z']]

fig, ax = plt.subplots(figsize=(10, max(6, 0.32 * len(lollipop_df))))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

y_pos = np.arange(len(lollipop_df))

# Stems
ax.hlines(
    y=y_pos, xmin=0, xmax=lollipop_df['pca_z'],
    color=bar_colors, linewidth=1.8, zorder=2
)

# Lollipop heads
ax.scatter(
    lollipop_df['pca_z'], y_pos,
    color=bar_colors, s=70, zorder=3, linewidth=0
)

ax.axvline(0, color=COLOR_GRID, linewidth=1.4, zorder=1)
ax.grid(axis='x', color=COLOR_GRID, linewidth=0.7, zorder=0)
ax.set_axisbelow(True)

ax.set_yticks(y_pos)
ax.set_yticklabels(lollipop_df['dist_name_short'], fontsize=9)

x_pad = lollipop_df['pca_z'].abs().max() * 0.18
ax.set_xlim(lollipop_df['pca_z'].min() - x_pad, lollipop_df['pca_z'].max() + x_pad)

for y, v in zip(y_pos, lollipop_df['pca_z']):
    label_color = COLOR_OVERPERFORM if v >= 0 else COLOR_UNDERPERFORM
    ha = 'left' if v >= 0 else 'right'
    offset = x_pad * 0.12 if v >= 0 else -x_pad * 0.12
    ax.text(
        v + offset, y, f"{v:+.2f}",
        va='center', ha=ha, fontsize=8, color=label_color, weight='bold', zorder=4
    )

ax.set_title("20 most overperforming and underperforming districts",
             fontsize=13, weight='bold', color=COLOR_TEXT, pad=14, loc='left')
ax.set_xlabel("Value-added performance (z-score)", fontsize=9.5, color=COLOR_TEXT, labelpad=10)

for spine in ['top', 'right', 'left']:
    ax.spines[spine].set_visible(False)

fig.suptitle(
    "District performance relative to socioeconomic expectations",
    fontsize=15, weight='bold', color=COLOR_TEXT, y=1.02, x=0.01, ha='left'
)
fig.text(
    0.01, 0.965,
    f"Distance from 0 represents how much they deviate from the model's expectations. Positive means overperformance, negative is underperformance",
    fontsize=10, color=COLOR_TEXT, ha='left', wrap=True
)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("performance_cluster.png", dpi=300, bbox_inches='tight')
plt.show()

accessible_tables = {
    "Beat-the-Odds": (
        top_over[['dist_name', 'ses_index', 'pca_z']]
        .rename(columns={
            'dist_name': 'District',
            'ses_index': 'SES index',
            'pca_z': 'Value-added z-score'
        })
        .round(2)
    ),
    "Underutilized": (
        top_under[['dist_name', 'ses_index', 'pca_z']]
        .rename(columns={
            'dist_name': 'District',
            'ses_index': 'SES index',
            'pca_z': 'Value-added z-score'
        })
        .round(2)
    ),
}



print("\n" + "="*60)
print("ACCESSIBLE DATA TABLE — Beat-the-Odds (overperforming, low-SES)")
print("="*60)
print(accessible_tables["Beat-the-Odds"].to_string(index=False))

print("\n" + "="*60)
print("ACCESSIBLE DATA TABLE — Underutilized (underperforming, high-SES)")
print("="*60)
print(accessible_tables["Underutilized"].to_string(index=False))

# endregion

# endregion
# %%

