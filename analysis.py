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
    'CAASPP-Math Standard Exceeded or Met (Levels 3 and 4) (District)': 'caaspp_math',
    'CAASPP-ELA Standard Exceeded or Met (Levels 3 and 4) (District)': 'caaspp_ela',
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
final['enroll_total']      = np.log(final['enroll_total'])

final = final[np.exp(final['enroll_total']) > 30]

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
    'avg_yrs_teaching', 'pct_frpm', 'pct_exp_instruction', 'suspension_rate', 
    'pct_chronic_absent', 'bach_pct', 'median_income', 'pct_el', 'diversity_idx', 
    'pct_swd', 'enroll_total', 'unemployment_pct', 'pct_first_yr_teachers', 
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
w = KNN.from_array(coords, k=8)
w_dist = KNN.from_array(coords, k=8, distance_metric='euclidean')

for i, neighbors in w_dist.neighbors.items():
    distances = [w_dist.weights[i][j] for j in range(len(neighbors))]
    w.weights[i] = [1.0 / (d + 0.0001) for d in distances]
    
w.transform = 'r'

formula = (
    '{outcome} ~ avg_yrs_teaching + pct_frpm + suspension_rate '
    '+ pct_chronic_absent + bach_pct + median_income + pct_el + diversity_idx '
    '+ locale + pct_swd + enroll_total'
)

print("\n--- Moran's I on OLS Baseline (IDW k=8) ---")
for outcome in ['caaspp_ela', 'caaspp_math']:
    res = smf.ols(formula.format(outcome=outcome), data=df_spatial).fit()
    df_spatial[f'ols_resid_{outcome}'] = res.resid.values
    moran = Moran(res.resid.values, w)
    print(f"{outcome}: R² = {res.rsquared:.4f}, I = {moran.I:.4f}, p = {moran.p_sim:.4f}, z = {moran.z_norm:.4f}")

# endregion

# region FMA specs

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from scipy.stats import zscore

SPECS = {
    'maximalist_all_factors': [
        'median_income', 'bach_pct', 'unemployment_pct', 'locale',
        'pct_frpm', 'pct_el', 'pct_swd', 'diversity_idx',
        'avg_yrs_teaching', 'pct_first_yr_teachers', 'pupil_teach_ratio', 'teaching_days',
        'enroll_total', 'exp_per_ada', 'pct_exp_instruction', 'pct_exp_admin',
        'suspension_rate', 'pct_chronic_absent', 'pct_exp_pupil_svc'
    ],
    
    'school_production_function': [
        'pct_frpm', 'pct_el', 'pct_swd', 'diversity_idx', 'enroll_total', 'locale',
        'avg_yrs_teaching', 'pct_first_yr_teachers', 'pupil_teach_ratio', 'teaching_days',
        'suspension_rate', 'pct_chronic_absent',
        'pct_exp_instruction', 'pct_exp_admin', 'pct_exp_pupil_svc'
    ],
    
    'structural_socioeconomic_resource': [
        'median_income', 'bach_pct', 'unemployment_pct', 'locale', 'enroll_total',
        'pct_frpm', 'pct_el', 'pct_swd',
        'exp_per_ada', 'pct_exp_instruction', 'pct_exp_admin',
        'avg_yrs_teaching', 'pupil_teach_ratio', 'teaching_days'
    ],
    
    'student_composition_and_climate': [
        'pct_frpm', 'pct_el', 'pct_swd', 'diversity_idx', 'enroll_total', 'locale',
        'median_income', 'unemployment_pct',
        'suspension_rate', 'pct_chronic_absent', 'pct_exp_pupil_svc',
        'pupil_teach_ratio', 'avg_yrs_teaching', 'pct_first_yr_teachers'
    ],
    
    'finance_and_human_capital': [
        'pct_frpm', 'pct_el', 'enroll_total', 'locale', 'bach_pct',
        'exp_per_ada', 'pct_exp_instruction', 'pct_exp_admin', 'pct_exp_pupil_svc',
        'avg_yrs_teaching', 'pct_first_yr_teachers', 'pupil_teach_ratio', 'teaching_days'
    ]
}

for key in SPECS:
    SPECS[key] = [col for col in SPECS[key] if col in df_spatial.columns]

# endregion

# region FMA random forest

cv = KFold(n_splits=5, shuffle=True, random_state=42)

rf_base = RandomForestRegressor(
    n_estimators=1000,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

outcomes     = ['caaspp_ela', 'caaspp_math']
spec_results = {outcome: {} for outcome in outcomes}

print("\n" + "="*65)
print("FMA: Fitting specs across outcomes")
print("="*65)

for outcome in outcomes:
    print(f"\n── {outcome} ──")
    for spec_name, features in SPECS.items():
        rf_data = final[features + [outcome, 'dist_name']].copy()
        rf_data = rf_data.dropna(subset=[outcome])
        
        if 'locale' in rf_data.columns:
            rf_data = pd.get_dummies(rf_data, columns=['locale'], dtype=float, drop_first=True)

        feature_cols = [c for c in rf_data.columns if c not in [outcome, 'dist_name']]
        X = rf_data[feature_cols]
        y = rf_data[outcome]

        num_cols   = X.select_dtypes(include='number').columns
        preprocess = ColumnTransformer([
            ('iterative', IterativeImputer(
                estimator=BayesianRidge(),
                max_iter=10,
                random_state=42,
                imputation_order='ascending'
            ), num_cols),
        ])

        pipe      = Pipeline([('prep', preprocess), ('rf', rf_base)])
        r2_scores = cross_val_score(pipe, X, y, cv=cv, scoring='r2')
        mean_r2   = r2_scores.mean()
        y_pred    = cross_val_predict(pipe, X, y, cv=cv)

        spec_results[outcome][spec_name] = {
            'resid': pd.Series(y.values - y_pred, index=rf_data.index),
            'r2':    mean_r2,
        }

        print(f"  {spec_name:<18}  CV R²: {mean_r2:.4f}  (n={len(y)})")

# endregion

# region FMA weights and residual aggregation

fma_resids   = {}
spec_weights = {}

print("\n" + "="*65)
print("FMA weights (proportional to CV R², floor 0)")
print("="*65)

for outcome in outcomes:
    specs       = spec_results[outcome]
    raw_weights = {k: max(v['r2'], 0.0) for k, v in specs.items()}
    total_w     = sum(raw_weights.values())

    if total_w == 0:
        norm_weights = {k: 1 / len(specs) for k in specs}
        print(f"  WARNING: all specs have R²≤0 for {outcome}; using equal weights")
    else:
        norm_weights = {k: w / total_w for k, w in raw_weights.items()}

    spec_weights[outcome] = norm_weights

    print(f"\n  {outcome}")
    for spec_name, wt in norm_weights.items():
        print(f"    {spec_name:<18}  weight: {wt:.4f}  (R²={specs[spec_name]['r2']:.4f})")

    resid_df     = pd.DataFrame({k: specs[k]['resid'] for k in specs})
    weighted_sum = pd.Series(0.0, index=resid_df.index)
    effective_wt = pd.Series(0.0, index=resid_df.index)

    for spec_name, wt in norm_weights.items():
        col  = resid_df[spec_name]
        mask = col.notna()
        weighted_sum[mask] += wt * col[mask]
        effective_wt[mask] += wt

    fma_resid              = weighted_sum / effective_wt.replace(0, np.nan)
    fma_resids[outcome]    = fma_resid.rename(f'fma_resid_{outcome}')

    print(f"\n  FMA residual coverage for {outcome}:")
    print(f"    Full:    {(effective_wt == 1.0).sum()}")
    print(f"    Partial: {((effective_wt > 0) & (effective_wt < 1.0)).sum()}")
    print(f"    None:    {(effective_wt == 0).sum()}")

# endregion

# region emp bayes

eb_resids = {}
print("\n" + "="*50)
print("EMPIRICAL BAYES: CLUSTERED SHRINKAGE")
print("="*50)

for outcome in ['caaspp_ela', 'caaspp_math']:
    res = smf.ols(formula.format(outcome=outcome), data=df_spatial).fit()
    df_spatial[f'ols_resid_{outcome}'] = res.resid.values
    
    eb_resid_series = pd.Series(index=df_spatial.index, dtype=float)
    
    # Calculate shrinkage independently for locale type
    for cluster_name, group in df_spatial.groupby('locale'):
        resid = group[f'ols_resid_{outcome}'].values
        n_students = np.exp(group['enroll_total'].values)
        
        p_bar = group[outcome].mean()
        
        v_j = (p_bar * (100 - p_bar)) / n_students
        
        total_var = np.var(resid, ddof=1) if len(resid) > 1 else 0
        mean_v_j = np.mean(v_j)
        tau_squared = max(0, total_var - mean_v_j)
        
        w_j = tau_squared / (tau_squared + v_j)
        eb_resid_series.loc[group.index] = w_j * resid
        
    eb_resids[outcome] = eb_resid_series.rename(f'eb_resid_{outcome}')
    print(f"  -> {outcome.upper()}: Clustered EB applied across {df_spatial['locale'].nunique()} locales.")

# endregion

# region performance ranking (SAR geosmoothed + winsorized)

import spreg
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore
from libpysal.weights import KNN, lag_spatial

combined = df_spatial[['dist_name', 'county']].copy()

for outcome in ['caaspp_ela', 'caaspp_math']:
    combined[f'ols_resid_{outcome}'] = df_spatial[f'ols_resid_{outcome}'].values
    combined = combined.join(fma_resids[outcome].rename(f'fma_resid_{outcome}'), how='left')
    combined = combined.join(eb_resids[outcome].rename(f'eb_resid_{outcome}'), how='left')

combined['enroll_total_raw'] = np.exp(df_spatial.loc[df_spatial['dist_name'].isin(combined['dist_name']), 'enroll_total'].values)
min_enrollment = 100
combined = combined[combined['enroll_total_raw'] >= min_enrollment].copy()

combined = combined.merge(df_spatial[['dist_name', 'lat', 'lon']], on='dist_name', how='left')

combined['fma_ela_z']  = zscore(combined['fma_resid_caaspp_ela'],  nan_policy='omit')
combined['fma_math_z'] = zscore(combined['fma_resid_caaspp_math'], nan_policy='omit')
combined['eb_ela_z']   = zscore(combined['eb_resid_caaspp_ela'],   nan_policy='omit')
combined['eb_math_z']  = zscore(combined['eb_resid_caaspp_math'],  nan_policy='omit')

print("\n" + "="*50)
print("PERFORMANCE RANKING: DYNAMIC SAR SMOOTHING + WINSORIZED PCA")
print("="*50)

components = ['fma_ela_z', 'fma_math_z', 'eb_ela_z', 'eb_math_z']

coords = combined[['lon', 'lat']].values
w_sub = KNN.from_array(coords, k=8)
w_sub.transform = 'r' 

X = np.ones((len(combined), 1))

for col in components:
    y = combined[col].values.reshape(-1, 1)
    
    sar_model = spreg.ML_Lag(y, X, w=w_sub, name_y=col)
    

    rho = max(0.0, min(sar_model.rho, 0.5))
    
    blend_neighbors = rho
    blend_self = 1.0 - rho
    
    neighbor_avg = lag_spatial(w_sub, combined[col].values)
    combined[col] = (blend_self * combined[col]) + (blend_neighbors * neighbor_avg)
    
    print(f"  -> SAR estimated split for {col}: {blend_self*100:.1f}% Self / {blend_neighbors*100:.1f}% Neighbors (rho = {sar_model.rho:.3f})")

for col in components:
    lower_bound = combined[col].quantile(0.01)
    upper_bound = combined[col].quantile(0.99)
    combined[col] = combined[col].clip(lower=lower_bound, upper=upper_bound)

print(f"  -> Winsorization applied (1st/99th percentiles).")

pca = PCA(n_components=1)
pca_raw = pca.fit_transform(combined[components])

combined['pca_z'] = (pca_raw - pca_raw.mean()) / pca_raw.std()

if combined[components[0]].corr(combined['pca_z']) < 0:
    combined['pca_z'] *= -1

print(f"  -> PCA Component 1 Variance Explained: {pca.explained_variance_ratio_[0]*100:.1f}%")

print("\n--- Structural overperformers (SAR Smoothed + Winsorized PCA) ---")
print(combined.sort_values('pca_z', ascending=False).head(20)[['dist_name', 'pca_z']])

print("\n--- Structural underperformers (SAR Smoothed + Winsorized PCA) ---")
print(combined.sort_values('pca_z').head(20)[['dist_name', 'pca_z']])

# endregion

# region spec weight diagnostics

print("\n" + "="*65)
print("Spec weight summary across outcomes")
print("="*65)
weight_df = pd.DataFrame(spec_weights).rename_axis('spec')
weight_df['avg_weight'] = weight_df.mean(axis=1)
print(weight_df.sort_values('avg_weight', ascending=False).round(4).to_string())

# endregion

# region spec stability diagnostic

print("\n" + "="*65)
print("Spec stability (residual SD across specs per district)")
print("="*65)

for outcome in outcomes:
    specs    = spec_results[outcome]
    resid_df = pd.DataFrame({k: specs[k]['resid'] for k in specs})
    combined[f'spec_sd_{outcome}'] = resid_df.std(axis=1)

    flagged = (
        combined[['dist_name', 'pca_z', f'spec_sd_{outcome}']]
        .dropna()
        .sort_values(f'spec_sd_{outcome}', ascending=False)
        .head(15)
    )
    print(f"\n  {outcome} — highest cross-spec residual SD")
    print(flagged.to_string(index=False))

# endregion

# region validation

import seaborn as sns
import shap

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
print("CHECK 2: SHAP VALUE ANALYSIS (ELA Model)")
print("="*50)

# We will use the maximalist model for the SHAP explanation
global_features = SPECS['maximalist_all_factors']
rf_data = final[global_features + ['caaspp_ela', 'dist_name']].dropna(subset=['caaspp_ela'])

if 'locale' in rf_data.columns:
    rf_data = pd.get_dummies(rf_data, columns=['locale'], drop_first=True, dtype=float)

feature_cols = [c for c in rf_data.columns if c not in ['caaspp_ela', 'dist_name']]
X = rf_data[feature_cols]
y = rf_data['caaspp_ela']

num_cols = X.select_dtypes(include='number').columns
preprocess = ColumnTransformer([
    ('iterative', IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=10,
        random_state=42,
        imputation_order='ascending'
    ), num_cols),
])

# Fit the preprocessing step
X_imputed = preprocess.fit_transform(X)
X_imputed_df = pd.DataFrame(X_imputed, columns=num_cols)

# Define and fit the Random Forest on the full dataset
rf_shap = RandomForestRegressor(n_estimators=1000, max_features='sqrt', random_state=42, n_jobs=-1)
rf_shap.fit(X_imputed_df, y)

# Calculate SHAP values
explainer = shap.TreeExplainer(rf_shap)
shap_values = explainer.shap_values(X_imputed_df)

# Plot summary
plt.figure(figsize=(12, 8))
plt.title("SHAP Feature Importance (Random Forest - ELA)")
shap.summary_plot(shap_values, X_imputed_df, feature_names=num_cols, show=False)
plt.tight_layout()
plt.savefig("shap_summary_ela.png")
plt.show()


print("\n" + "="*50)
print("CHECK 3: SUBGROUP BIAS AND VARIANCE CHECK")
print("="*50)

check_df = combined.merge(
    df_spatial[['dist_name', 'enroll_total', 'locale', 'dist_type', 'pct_frpm']], 
    on='dist_name', 
    how='left'
)

check_df['enrollment_raw'] = np.exp(check_df['enroll_total'])

check_df['size_quintile'] = pd.qcut(
    check_df['enrollment_raw'], 
    q=5, 
    labels=['Smallest', 'Small', 'Medium', 'Large', 'Largest']
)

# 1. Check for systematic bias 
print("\n--- Mean PCA Z-Score by District Size ---")
print(check_df.groupby('size_quintile', observed=False)['pca_z'].mean().round(3))

print("\n--- Mean PCA Z-Score by Locale ---")
print(check_df.groupby('locale', observed=False)['pca_z'].mean().round(3))

# 2. Check for over-shrinkage by size
print("\n--- Variance of PCA Z-Score by District Size ---")
print(check_df.groupby('size_quintile', observed=False)['pca_z'].var().round(3))

# 3. Check correlation with Poverty
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
print("STABILITY 2: MONTE CARLO RANK SIMULATION (EXACT COV + VARIANCE POOLING)")
print("="*50)

corr_matrix = combined[components].corr()
N = len(components)
avg_r = (corr_matrix.values.sum() - N) / (N**2 - N)
shrinkage_factor = np.sqrt((1 + (N - 1) * avg_r) / N)

raw_se = combined[components].std(axis=1) * shrinkage_factor
median_se = raw_se.median()


combined['se_z'] = (0.5 * raw_se) + (0.5 * median_se)
combined['se_z'] = combined['se_z'].fillna(median_se)

print(f"  -> Average global correlation: {avg_r:.3f}")
print(f"  -> Covariance shrinkage factor: {shrinkage_factor:.3f}")
print(f"  -> Variance Pooling: Blended individual SE with global median SE ({median_se:.3f})")

n_sims = 100000
n_districts = len(combined)
simulated_ranks = np.zeros((n_districts, n_sims))

for i in range(n_sims):
    noise = np.random.normal(loc=0, scale=combined['se_z'])
    simulated_score = combined['pca_z'] + noise
    simulated_ranks[:, i] = simulated_score.rank(ascending=False, method='min')

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

# Grab the 2 districts above the median, the median itself, and the 2 below it
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

# Prevent Jupyter duplicate column errors on cell re-runs
if 'assist_status' in combined.columns:
    combined = combined.drop(columns=['assist_status'])

combined = combined.merge(df_spatial[['dist_name', 'assist_status']], on='dist_name', how='left')

# pca_z is already standardized natively, so we just pass it through directly
combined['final_z_standardized'] = combined['pca_z']

def assign_tier(z, spread):
    if spread > 400:
        return "Mixed/Volatile Results"
    
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

import scipy.stats as stats
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


# region PLOTS

# region SHAP Bar Plot

print("\n" + "="*50)
print("INTERPRETABILITY: SHAP BAR PLOT")
print("="*50)

plt.figure(figsize=(10, 6))
plt.title("Top Factors Driving ELA Scores (Average Impact)")

shap.summary_plot(
    shap_values, 
    X_imputed_df, 
    plot_type="bar", 
    show=False
)

plt.tight_layout()
plt.savefig("shap_bar_ela.png")
plt.show()
# endregion

# region ICE Plots (Individual Conditional Expectation)

from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt

print("\n" + "="*60)
print("INTERPRETABILITY: 6-VARIABLE ICE GRID FOR POLICY")
print("="*60)

features_to_plot = [
    'pct_frpm', 'pct_el', 'avg_yrs_teaching', 
    'suspension_rate', 'pct_chronic_absent', 'exp_per_ada'
]

label_mapping = {
    'pct_frpm': 'Student Poverty (%)',
    'pct_el': 'English Learners (%)',
    'avg_yrs_teaching': 'Avg Teaching Experience (Years)',
    'suspension_rate': 'Suspension Rate (%)',
    'pct_chronic_absent': 'Chronic Absenteeism (%)',
    'exp_per_ada': 'Funding per Student ($ per ADA)'
}

y_limit_min = 20  
y_limit_max = 80  

fig, ax = plt.subplots(figsize=(16, 10))

display = PartialDependenceDisplay.from_estimator(
    estimator=rf_shap,
    X=X_imputed_df,
    features=features_to_plot,
    kind='both',
    subsample=100,
    random_state=42,
    grid_resolution=50,
    ice_lines_kw={"color": "tab:blue", "alpha": 0.08, "linewidth": 0.5},
    pd_line_kw={"color": "tab:orange", "linewidth": 3.5, "alpha": 1},
    ax=ax
)

for i, feature_name in enumerate(features_to_plot):
    sub_ax = display.axes_.flatten()[i]
    if sub_ax is not None:
        sub_ax.set_xlabel(label_mapping[feature_name], fontsize=11, weight='bold')
        sub_ax.set_ylabel("Predicted % Meeting ELA Standards", fontsize=9)
        sub_ax.set_ylim(y_limit_min, y_limit_max)
        sub_ax.grid(True, linestyle='--', alpha=0.5)

fig.suptitle("Isolated Impact Curves: Systemic Trends (Orange) vs. Individual Districts (Blue)", 
             fontsize=18, weight='bold', y=0.96)

plt.subplots_adjust(top=0.88, bottom=0.08, wspace=0.3, hspace=0.35)
plt.savefig("policy_ice_grid_ela.png", dpi=300)
plt.show()

# endregion

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

CMAP_OVER  = plt.cm.Greens

CMAP_UNDER = plt.cm.Reds_r 

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["text.color"] = COLOR_TEXT
plt.rcParams["axes.edgecolor"] = COLOR_TEXT
plt.rcParams["axes.labelcolor"] = COLOR_TEXT
plt.rcParams["xtick.color"] = COLOR_TEXT
plt.rcParams["ytick.color"] = COLOR_TEXT

N_LABELS = 6


panel_specs = {
    "Beat-the-Odds": dict(
        group="Beat-the-Odds",
        point_color="#14532D",   # Solid deep green
        label_color=COLOR_OVERPERFORM,
        title="Low-SES districts that outperform expectations",
        sort_ascending=False,
    ),
    "Underutilized": dict(
        group="Underutilized",
        point_color="#7A271A",   # Solid deep red
        label_color=COLOR_UNDERPERFORM,
        title="High-SES districts that underperform expectations",
        sort_ascending=True,
    ),
}

fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
fig.patch.set_facecolor("white")

accessible_tables = {}

for ax, (panel_name, spec) in zip(axes, panel_specs.items()):
    grp = spec["group"]
    panel_df = cluster_df[cluster_df['strategic_group'] == grp]
    background_df = cluster_df[cluster_df['strategic_group'] != grp]

    ax.set_facecolor("white")
    ax.axhline(0, color=COLOR_GRID, linewidth=1.2, zorder=1)
    ax.axvline(0, color=COLOR_GRID, linewidth=1.2, zorder=1)
    ax.grid(True, color=COLOR_GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Plot background points in a faint gray color for clean styling
    ax.scatter(
        background_df['ses_index'], background_df['pca_z'], 
        color="#E5E7EB", alpha=0.6, s=26, linewidth=0, zorder=2
    )

    # Plot active points with a clean, solid color hex code
    ax.scatter(
        panel_df['ses_index'], panel_df['pca_z'], 
        color=spec["point_color"], alpha=0.9, s=50, linewidth=0, zorder=3
    )

    # Note: Colorbar instantiation and formatting blocks have been removed entirely

    # Set boundaries BEFORE adjust_text so label positioning respects axis constraints
    if grp == "Beat-the-Odds":
        x_lo = panel_df['ses_index'].min() - 0.3
        x_hi = 0.0  
        y_lo = 0.0  
        y_hi = panel_df['pca_z'].max() + 0.5
    else:
        x_lo = 0.0  
        x_hi = panel_df['ses_index'].max() + 0.3
        y_lo = panel_df['pca_z'].min() - 0.5
        y_hi = 0.0  

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    top_labels = panel_df.sort_values('pca_z', ascending=spec["sort_ascending"]).head(N_LABELS)

    texts = []
    label_x, label_y = [], []
    for _, row in top_labels.iterrows():
        t = ax.text(
            row['ses_index'], row['pca_z'], row['dist_name_short'],
            fontsize=8, color=spec["label_color"], weight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=1.2),
            zorder=6
        )
        texts.append(t)
        label_x.append(row['ses_index'])
        label_y.append(row['pca_z'])

    adjust_text(
        texts,
        ax=ax,
        x=label_x,
        y=label_y,
        arrowprops=dict(arrowstyle='-', color=COLOR_TEXT, lw=0.6, alpha=0.6),
        expand=(1.2, 1.4),
        force_text=(0.6, 0.8),
        max_move=30,
        lim=2000,
    )

    ax.set_title(spec["title"], fontsize=13, weight='bold', color=COLOR_TEXT, pad=14, loc='left')
    ax.set_xlabel("Socioeconomic status index (higher = wealthier)", fontsize=9.5, color=COLOR_TEXT, labelpad=10)
    ax.set_ylabel("Value-added performance (z-score)", fontsize=9.5, color=COLOR_TEXT, labelpad=10)

    accessible_tables[grp] = (
        top_labels[['dist_name', 'ses_index', 'pca_z']]
        .rename(columns={
            'dist_name': 'District',
            'ses_index': 'SES index',
            'pca_z': 'Value-added z-score'
        })
        .round(2)
    )


fig.suptitle(
    "Performance relative to socioeconomic expectations, by district",
    fontsize=15, weight='bold', color=COLOR_TEXT, y=1.04, x=0.01, ha='left'
)
fig.text(
    0.01, 0.965,
    "Color intensity shows how far each district's performance deviates from what its "
    "socioeconomic profile would predict. Faint gray points are districts outside this panel's group.",
    fontsize=10, color=COLOR_TEXT, ha='left', wrap=True
)

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.savefig("performance_cluster.png")
plt.show()



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
