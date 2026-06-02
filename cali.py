import pandas as pd

ca = pd.read_excel("C:\\Users\\CYBER_SLAVE\\Downloads\\423 Project\\CA School Analysis\\data\\fiscalyear2024to25.xlsx", header = 3)

ca.drop([58, 59, 60, 61, 62], axis = 0)

# saved table id = 658116, fiscal year 24-25

districts = pd.read_csv("C:\\Users\\CYBER_SLAVE\\Downloads\\423 Project\\CA School Analysis\\data\\NCES.csv", header = 3)

districts.drop([2110, 2111, 2112, 2113, 2114], axis = 0)

compare = pd.read_csv("C:\\Users\\CYBER_SLAVE\\Downloads\\423 Project\\CA School Analysis\\data\\ED-Data.csv")


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

compare['District Name Clean'] = compare['District Name Clean'].replace(manual_map)

merged = compare.merge(districts, left_on='District Name Clean', right_on='Agency Name Clean', how='left')
print(merged['Agency Name Clean'].isna().sum())

districts['Agency Type [District] 2024-25'].value_counts()