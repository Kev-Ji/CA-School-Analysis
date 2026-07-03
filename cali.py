import pandas as pd

ca = pd.read_excel("data/fiscalyear2024to25.xlsx", header=3)

ca.drop([58, 59, 60, 61, 62], axis=0)

# saved table id = 658116, fiscal year 24-25

districts = pd.read_csv("data/NCES.csv", header=3)
# source: https://nces.ed.gov/ccd/elsi/

districts.drop([2110, 2111, 2112, 2113, 2114], axis=0)

compare = pd.read_csv("data/ED-Data.csv")
# source: https://www.ed-data.org/District/Humboldt/Scotia-Union-Elementary 


compare['District Name Clean'] = (compare['District Name']
    .str.upper()
    .str.replace(r'\(.*?\)', '', regex=True)
    .str.replace(r'^SBE - ', '', regex=True)
    .str.replace(r'^SBC - ', '', regex=True)
    .str.replace(',', '')
    .str.strip()
)

districts['Agency Name Clean'] = (districts['Agency Name']
    .str.upper()
    .str.replace(r'\s+DISTRICT$', '', regex=True)
    .str.strip()
)

manual_map = {
    'BAYPOINT PREPARATORY ACADEMY SAN DIEGO': 'BAYPOINT PREPARATORY ACADEMY - SAN DIEGO',
    'KERN COUNTY SUPERINTENDENT OF SCHOOLS': 'KERN COUNTY OFFICE OF EDUCATION',
    'ROSEVILLE CITY': 'ROSEVILLE CITY ELEMENTARY'
}

compare['District Name Clean'] = compare['District Name Clean'].replace (manual_map)

merged = compare.merge(districts, left_on='District Name Clean', right_on='Agency Name Clean', how='left')
print(merged['Agency Name Clean'].isna().sum())

districts['Agency Type [District] 2024-25'].value_counts()

merged.to_csv("data/publicCA.csv")


merged = merged.drop_duplicates(subset = ['District Name'], keep = 'first')

merged = merged[
    ~merged["District Type (District)"].isin([
        "State Board of Education Charter",
        "Common Administration District",
        "Statewide Benefit Charter"
    ])
]

merged['District Type (District)'].value_counts()

import matplotlib.pyplot as plt 

merged['CAASPP-Math Standard Exceeded or Met (Levels 3 and 4) (District)'] = pd.to_numeric(
    merged['CAASPP-Math Standard Exceeded or Met (Levels 3 and 4) (District)'],
    errors="coerce"
)

## this checks if free/reduced lunch % is behaving normally 

plt.hist(merged['Free/Reduced Meals % (District)'], bins = 30)

# The above was combining performance and district spending data. Below will be getting demographics of these school districts such as parental educ. and median income. 

# Education tabulation source: https://nces.ed.gov/programs/edge/Demographic/ACS
# I need: educational attainment, percentage of parents below poverty level, and income and benefits, employment status too seems reasonable 

#DP02_65pct - educational attainment (bachelors)
#DP03_62est - median income + benefits 
#DP03_62moe - median income + benefits MoE
#DP03_9pct - unemployment rate 
#DP03_119pct - PERCENTAGE OF FAMILIES AND PEOPLE WHOSE INCOME IN THE PAST 12 MONTHS IS BELOW POVERTY LEVEL 

check = pd.read_csv(r"data/ACS-ED tabulation/genpop.txt", delimiter="|")

educ = pd.read_csv(r"data/ACS-ED tabulation/parent-educ.txt", delimiter = "|")

check['bach_pct'] = educ['DP02_65pct']

var = ['GeoId', 'Geography', 'LEAID', 'DP03_62est', 'DP03_62moe', 'DP03_9pct', 'DP03_119pct', 'bach_pct']

econ = check[var].rename(columns={
    'DP03_62est': 'median_income',
    'DP03_62moe': 'median_income_moe',
    'DP03_9pct':  'unemployment_pct',
    'DP03_119pct': 'poverty_pct'
})

econ['Geography'] = econ['Geography'].str.replace(' School District, CA', '', regex = False)

assist = pd.read_csv("data/California_School_District_Areas_2024-25.csv")

# source: https://data.ca.gov/dataset/california-school-district-areas-2024-25

merge2 = econ.merge(assist, left_on='LEAID', right_on='FedID', how='left')

merge_clean = merge2[['FedID', 'median_income', 'median_income_moe', 'unemployment_pct', 'poverty_pct', 'bach_pct', 'AssistStatus', 'SEDpct']]

final = merged.merge(merge2, left_on = 'Agency ID - NCES Assigned [District] Latest available year', right_on = 'FedID', how = 'left')

# region charter disentanglement

# region charter disentanglement

# 1. Load the CAASPP and Directory files
caaspp_raw = pd.read_csv("data/sb_ca2025_all_csv_v1.txt", sep='^', encoding='latin1', dtype={'County Code': str, 'District Code': str, 'School Code': str})
pubschls = pd.read_csv("data/pubschls.txt", sep='\t', encoding='latin1', dtype={'CDSCode': str})

# 2. Prepare keys for merging
caaspp_raw['County Code'] = caaspp_raw['County Code'].str.zfill(2)
caaspp_raw['District Code'] = caaspp_raw['District Code'].str.zfill(5)
caaspp_raw['School Code'] = caaspp_raw['School Code'].str.zfill(7)
caaspp_raw['CDSCode'] = caaspp_raw['County Code'] + caaspp_raw['District Code'] + caaspp_raw['School Code']
caaspp_raw['CDCode'] = caaspp_raw['County Code'] + caaspp_raw['District Code']

# 3. Merge charter status onto CAASPP data
caaspp_raw = caaspp_raw.merge(pubschls[['CDSCode', 'Charter']], on='CDSCode', how='left')

# 4. Clean numeric columns for aggregation
caaspp_raw['total_tested'] = pd.to_numeric(caaspp_raw['Total Students Tested with Scores'], errors='coerce')
caaspp_raw['total_pass'] = pd.to_numeric(caaspp_raw['Count Standard Met and Above'], errors='coerce')

# 5. Filter for target data and aggregate
caaspp_filtered = caaspp_raw[
    (caaspp_raw['School Code'] != '0000000') & 
    (caaspp_raw['Student Group ID'] == 1) &
    (caaspp_raw['Test ID'].isin([1, 2]))
].copy()

agg = caaspp_filtered.groupby(['CDCode', 'Test ID', 'Charter']).agg(
    dist_tested=('total_tested', 'sum'),
    dist_pass=('total_pass', 'sum')
).reset_index()

agg['pass_rate'] = (agg['dist_pass'] / agg['dist_tested']) * 100

# 6. Pivot into the 6 required columns
pivot_dis = agg.pivot_table(index='CDCode', columns=['Test ID', 'Charter'], values='pass_rate')
pivot_dis.columns = ['non_charter_ela_caaspp', 'charter_ela_caaspp', 'non_charter_math_caaspp', 'charter_math_caaspp']

# 7. Calculate combined totals
agg_comb = caaspp_filtered.groupby(['CDCode', 'Test ID'])[['total_pass', 'total_tested']].sum()
agg_comb['rate'] = (agg_comb['total_pass'] / agg_comb['total_tested']) * 100
pivot_comb = agg_comb.pivot_table(index='CDCode', columns='Test ID', values='rate')
pivot_comb.columns = ['combined_ela_caaspp', 'combined_math_caaspp']

# 8. Build NCES LEAID <-> CA CDCode crosswalk from pubschls
crosswalk = pubschls[['NCESDist', 'CDSCode']].copy()
crosswalk['CDCode'] = crosswalk['CDSCode'].str[:7]
crosswalk['NCESDist'] = crosswalk['NCESDist'].astype(str).str.split('.').str[0].str.zfill(7)
crosswalk = crosswalk.drop_duplicates(subset='NCESDist')
crosswalk = crosswalk.rename(columns={'CDCode': 'CDCode_xwalk'})  # avoid collision

final['NCESDist_join'] = (
    final['Agency ID - NCES Assigned [District] Latest available year']
    .astype(str)
    .str.split('.').str[0]
    .str.zfill(7)
)

final = final.merge(
    crosswalk[['NCESDist', 'CDCode_xwalk']],
    left_on='NCESDist_join', right_on='NCESDist', how='left'
)

# 9. Merge CAASPP metrics onto final using the crosswalked CDCode
final = final.merge(pivot_dis, left_on='CDCode_xwalk', right_index=True, how='left')
final = final.merge(pivot_comb, left_on='CDCode_xwalk', right_index=True, how='left')

final = final.drop(columns=['NCESDist_join', 'NCESDist'], errors='ignore')


# endregion

final.to_csv("data/final.csv")

# awards

all_award = pd.read_excel(r"data/dsaawards.xlsx", header=5)

award = all_award[all_award['Year'] > 2023]

award.to_csv("data/2024-25award.csv")

