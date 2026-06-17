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
)f

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

final.to_csv("data/final.csv")

