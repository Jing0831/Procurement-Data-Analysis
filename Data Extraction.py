# Import Package
from selenium import webdriver
import pandas as pd
from urllib.request import urlretrieve as retrieve

# Creat an empty list (RUN THE FOLLOWING CODE ONLY IN THE FIRST TIME)
date = []
num_row = []
update_list = pd.DataFrame({'Date': date, 'Row Number': num_row})
update_list.to_csv('update_list_log.csv', index = False)

# Start From here:
path = 'chromedriver.exe'
driver = webdriver.Chrome(path)
url = 'https://data.cityofnewyork.us/City-Government/Recent-Contract-Awards/qyyg-4tf5'
dataurl = 'https://data.cityofnewyork.us/api/views/qyyg-4tf5/rows.csv?accessType=DOWNLOAD'
driver.get(url)

new_date = driver.find_element_by_class_name("date").text
pagelabel = driver.find_element_by_class_name("pager-label").text.split()
new_rownumber = pagelabel[7]
update_list = pd.read_csv('update_list_log.csv')

new_list = pd.DataFrame({'Date': [new_date], 'Row Number': [new_rownumber]})
if new_rownumber not in update_list['Row Number'].tolist():
    update_list = pd.concat([new_list, update_list])
    update_list.to_csv('update_list_log.csv', index=False)
    retrieve(dataurl, new_date + '.csv')
else:
    pass

driver.close()

## Step3: Data Analytics




