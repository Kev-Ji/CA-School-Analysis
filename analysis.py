import numpy as np 
import pandas as pd
import statsmodels.api as sm 
import statsmodels.formula.api as smf 

# region data

final = pd.read_csv("data/final.csv")

final = merged[
    ~merged["District Type (District)"].isin([
        "State Board of Education Charter",
        "Common Administration District",
        "Statewide Benefit Charter"
    ])
]

final = final.rename(columns={
    'Unnamed: 0': 'idx',

    # --- Core district identifiers ---
    'District Name': 'dist_name',
    'County Name (District)': 'county',
    'District Type (District)': 'dist_type',
    'All Charter Schools (Y/N) (District)': 'all_charter',
    'Grade Span (District)': 'grade_span',
    'Zip (District)': 'zip',
    'City (District)': 'city',

    # --- Enrollment & demographics ---
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

    # --- Outcomes ---
    'Cohort Graduates % (District)': 'pct_grad',
    'Cohort Grads by Socio. Econ. DisAdvtg % (District)': 'pct_grad_sed',
    'CAASPP-Math Standard Exceeded or Met (Levels 3 and 4) (District)': 'caaspp_math',
    'CAASPP-ELA Standard Exceeded or Met (Levels 3 and 4) (District)': 'caaspp_ela',

    # --- Teacher characteristics ---
    'Student/Teacher Ratio (District)': 'stu_teach_ratio',
    '1st Year Teachers (District)': 'pct_first_yr_teachers',
    'Experienced Teachers (District)': 'pct_exp_teachers',
    'Teacher Service Days (District)': 'teacher_svc_days',
    'Teaching Days (District)': 'teaching_days',
    'Inexperienced Teachers (District)': 'pct_inexp_teachers',
    'Avg Years Teaching (District)': 'avg_yrs_teaching',

    # --- Climate & behavior ---
    'Suspension Rate (District)': 'suspension_rate',
    'Chronic Absenteeism % (District)': 'pct_chronic_absent',

    # --- Finance ---
    'Current Exp of Educ per ADA (Ed Code 41372) (District)': 'exp_per_ada',
    'Gen Fund Exp by Activity - 1000-1999 Instruction Exp % (District)': 'pct_exp_instruction',
    'Gen Fund Exp by Activity - 2000-2999 Instruc-related Svcs Exp % (District)': 'pct_exp_instruct_svc',
    'Gen Fund Exp by Activity - 3000-3999 Pupil Services Exp % (District)': 'pct_exp_pupil_svc',
    'Gen Fund Exp by Activity - 4000-4999 Ancillary Services Exp % (District)': 'pct_exp_ancillary',
    'Gen Fund Exp by Activity - 6000-6999 Enterprise Exp % (District)': 'pct_exp_enterprise',
    'Gen Fund Exp by Activity - 7000-7999 General Administration Exp % (District)': 'pct_exp_admin',
    'Gen Fund Exp by Activity - 8000-8999 Plant Services Exp % (District)': 'pct_exp_plant',

    # --- District name / ID variants ---
    'District Name Clean': 'dist_name_clean',
    'Agency Name': 'agency_name',
    'Agency Name Clean': 'agency_name_clean',
    'Agency Name [District] 2024-25': 'agency_name_2425',
    'Agency ID - NCES Assigned [District] Latest available year': 'nces_id',
    'State Agency ID [District] 2024-25': 'state_agency_id',

    # --- State / geo identifiers ---
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

    # --- NCES locale / status ---
    'Locale [District] 2024-25': 'locale',
    'Agency Type [District] 2024-25': 'agency_type',
    'Agency Level (SY 2017-18 onward) [District] 2024-25': 'agency_level',
    'LEA Charter Status [District] 2024-25': 'lea_charter_status',
    'Start of Year Status [District] 2024-25': 'status_start',
    'Updated Status [District] 2024-25': 'status_updated',
    'Effective Date of Updated Status [District] 2024-25': 'status_eff_date',
    'Web Site URL [District] 2024-25': 'website',

    # --- Metro / congressional geography ---
    'CBSA Name [District] 2024-25': 'cbsa_name',
    'CBSA ID [District] 2024-25': 'cbsa_id',
    'CSA Name [District] 2024-25': 'csa_name',
    'CSA ID [District] 2024-25': 'csa_id',
    'Metro Micro Area Code [District] 2024-25': 'metro_code',
    'Congressional Code [District] 2024-25': 'congressional_code',
    'Supervisory Union (ID) Number [District] 2024-25': 'supervisory_union_id',

    # --- Grade span ---
    'Lowest Grade Offered [District] 2024-25': 'grade_low',
    'Highest Grade Offered [District] 2024-25': 'grade_high',

    # --- Student counts (NCES) ---
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

    # --- Staff (NCES) ---
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

    # --- ACS / Census economic variables ---
    'median_income': 'median_income',
    'median_income_moe': 'income_moe',
    'unemployment_pct': 'unemployment_pct',
    'poverty_pct': 'poverty_pct',
    'bach_pct': 'bach_pct',

    # --- CA enrollment detail (CDCode source) ---
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
    'ELpct': 'pct_el_ca',       # disambiguate from pct_el above
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

    # --- District geometry ---
    'DistrctAreaSqMi': 'area_sq_mi',
    'Shape__Area': 'shape_area',
    'Shape__Length': 'shape_length',
})

locale_map = {
    '11-City: Large':      'Not Rural',
    '12-City: Mid-size':   'Not Rural',
    '13-City: Small':      'Not Rural',
    '21-Suburb: Large':    'Not Rural',
    '22-Suburb: Mid-size': 'Not Rural',
    '23-Suburb: Small':    'Not Rural',
    '31-Town: Fringe':     'Not Rural',
    '32-Town: Distant':    'Not Rural',
    '33-Town: Remote':     'Not Rural',
    '41-Rural: Fringe':    'Rural',
    '42-Rural: Distant':   'Rural',
    '43-Rural: Remote':    'Rural',
}

assist_map = {
    'Differentiated, Year 1':   'Differentiated',
    'Differentiated, Year 2':   'Differentiated',
    'General':                  'General'
}

final['assist_status'] = final['assist_status'].map(assist_map)

final['locale'] = final['locale'].map(locale_map)

final['caaspp_ela'] = pd.to_numeric(final['caaspp_ela'], errors='coerce')
final['median_income'] = pd.to_numeric(final['median_income'], errors='coerce')
final['pupil_teach_ratio'] = pd.to_numeric(final['pupil_teach_ratio'], errors='coerce')
final['fte_teachers'] = pd.to_numeric(final['fte_teachers'], errors='coerce')

# endregion

results = smf.ols('caaspp_ela ~ avg_yrs_teaching + pct_frpm + pct_exp_instruction + suspension_rate + pct_chronic_absent + bach_pct + median_income + pct_el + diversity_idx + locale + sed_pct + pct_swd + teaching_days', data = final).fit()
robust = results.get_robustcov_results(cov_type='HC3')

robust.summary()


from statsmodels.stats.outliers_influence import variance_inflation_factor

X = final[['avg_yrs_teaching', 'pct_frpm', 'suspension_rate',
           'pct_chronic_absent', 'bach_pct', 'median_income',
           'diversity_idx', 'pct_el']].dropna()

vif_data = pd.DataFrame({
    'Feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
}).sort_values('VIF', ascending=False)

print(vif_data)

# region spatial

from libpysal.weights import KNN
from esda.moran import Moran


model_cols = ['caaspp_ela', 'avg_yrs_teaching', 'pct_frpm', 'pct_exp_instruction',
              'suspension_rate', 'pct_chronic_absent', 'bach_pct', 'median_income',
              'pct_el', 'diversity_idx', 'locale', 'sed_pct', 'pct_swd', 'lat', 'lon']

df_spatial = final.dropna(subset=model_cols).copy()

# Fit on this clean subset
results = smf.ols(
    'caaspp_ela ~ avg_yrs_teaching + pct_frpm + pct_exp_instruction + suspension_rate '
    '+ pct_chronic_absent + bach_pct + median_income + pct_el + diversity_idx '
    '+ locale + sed_pct + pct_swd',
    data=df_spatial
).fit()

# Now residuals and rows are guaranteed to align
df_spatial['resid'] = results.resid.values

coords = list(zip(df_spatial['lon'], df_spatial['lat']))
w = KNN(coords, k=5)
w.transform = 'r'

moran = Moran(df_spatial['resid'], w)
print(f"Moran's I:  {moran.I:.4f}")
print(f"p-value:    {moran.p_sim:.4f}")
print(f"z-score:    {moran.z_norm:.4f}")

# endregion


# region RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

features = [
    'avg_yrs_teaching',
    'pct_frpm',
    'pct_exp_instruction',
    'suspension_rate',
    'pct_chronic_absent',
    'bach_pct',
    'median_income',
    'pct_el',
    'diversity_idx',
    'locale',
    'sed_pct',
    'pct_swd',
    'teaching_days'
]

rf_data = final[features + ['caaspp_math']].copy()
rf_data = rf_data.dropna(subset=['caaspp_math'])

X = rf_data[features]
y = rf_data['caaspp_math']

num_cols = X.select_dtypes(include='number').columns
cat_cols = ['locale']

preprocess = ColumnTransformer([
    ('num', SimpleImputer(strategy='median'), num_cols),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

rf = RandomForestRegressor(
    n_estimators=1000,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ('prep', preprocess),
    ('rf', rf)
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = cross_val_score(
    pipe,
    X,
    y,
    cv=cv,
    scoring='r2'
)

print("CV R²:", r2_scores.mean())
print("SD:", r2_scores.std())

# endregion

# region OLS CV

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

ols_pipe = Pipeline([
    ('prep', preprocess),
    ('ols', LinearRegression())
])

ols_cv = cross_val_score(
    ols_pipe,
    X,
    y,
    cv=10,
    scoring='r2'
)

print("OLS CV R²:", ols_cv.mean())

# endregion

### 
## RESID VS. OLS 
### 

