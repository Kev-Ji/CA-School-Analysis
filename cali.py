import pandas as pd

ca = pd.read_excel("data/fiscalyear2024to25.xlsx", header=3)



ca.drop([58, 59, 60, 61, 62], axis=0)

# saved table id = 658116, fiscal year 24-25

districts = pd.read_csv("data/NCES.csv", header=3)

districts.drop([2110, 2111, 2112, 2113, 2114], axis=0)

compare = pd.read_csv("data/ED-Data.csv")


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



