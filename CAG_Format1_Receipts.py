#!/usr/bin/env python
# coding: utf-8
import json
import pandas as pd
import os
import re
# In[1]:


# Function to check entire row for keywords
def check_keywords(row):
    if 'Receipts' in row.to_string():
        return 'R'
    elif 'Disbursements' in row.to_string():
        return 'D'
    return None


# In[2]:



def extract_number(filename):
    match = re.search(r'output_pg_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return 0  # default value if no match is found


# In[3]:



# Define the function to process one JSON file
def process_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    ocr_data = {
        'lines': data['imageText']['lines']
    }

    # Thresholds
    y_threshold = 0.01
    x_threshold = 0.03

    rows = []

    for line in ocr_data["lines"]:
        y_avg = sum([v["y"] for v in line["boundingPolygon"]["normalizedVertices"]]) / 4
        x_start = line["boundingPolygon"]["normalizedVertices"][0]["x"]
        added = False
        for row in rows:
            if abs(row["y_avg"] - y_avg) < y_threshold:
                added_to_cell = False
                for cell in row["cells"]:
                    if abs(cell["x_start"] - x_start) < x_threshold:
                        cell["texts"].append(line["text"])
                        added_to_cell = True
                        break
                if not added_to_cell:
                    row["cells"].append({"x_start": x_start, "texts": [line["text"]]})
                added = True
                break
        if not added:
            rows.append({"y_avg": y_avg, "cells": [{"x_start": x_start, "texts": [line["text"]]}]})

    # Convert rows to pandas DataFrame
    data = []
    for row in rows:
        out_row = []
        for cell in row["cells"]:
            out_row.append(" ".join(cell["texts"]))
        data.append(out_row)

    df = pd.DataFrame(data)

    return df


# In[6]:


def has_decimal_number(item):
    return bool(re.search(r'^\d+\.\d+$', item))


def process_row(row):
    # If there are decimal numeric values in the row, just return the current value
    if has_decimal_number(str(row[PYA])) or has_decimal_number(str(row[CYA])):
        return row['Code/Description']

    cols = row.index
    row_data = [str(item) for item in row if pd.notnull(item)]
    for idx, item in enumerate(reversed(row_data)):
        match = re.search(r'\b\d{4}\b', item)
        if match:
            idx_of_match = item.rfind(match.group())
            concatenated_str = item[idx_of_match:]
            if idx != 0:
                concatenated_str += ' ' + ' '.join(row_data[-idx:])
            return concatenated_str
    return row['Code/Description']

#find heading for the total and add it into a separate column for all the rows contained in it

def shift_values(row):
    # Check if Code/Description is None
    if pd.isna(row['Code/Description']):
        # Search for the pattern of a 4-digit code from the end of the string
        match = re.search(r'(\d{4}-.*$)', row['HindiCode'])
        if match:
            # Assign everything including and after the 4-digit code to Code/Description
            row['Code/Description'] = match.group(1)
            # Extract everything before the 4-digit code in HindiCode
            row['HindiCode'] = row['HindiCode'].rsplit(match.group(1), 1)[0].strip()
    return row


# Pattern to detect headers, subheaders, and sub-subheaders
pattern = r'(?:[A-Z]-|\([a-z]\)|\([ivx]+\))'


def has_numeric(row):
    return bool(re.search(r'^\d+(\.\d+)?$', str(row[PYA]))) and bool(re.search(r'^\d+(\.\d+)?$', str(row[CYA])))


def process_row(row):
    # If there are numeric values in the row, just return the current value
    if has_numeric(row) or ("TOTAL" in str(row['Code/Description'])):
        return row['Code/Description']

    row_data = [str(item) for item in row if pd.notnull(item)]
    for idx, item in enumerate(reversed(row_data)):
        match = re.search(pattern, item)
        if match:
            idx_of_match = item.rfind(match.group())
            concatenated_str = item[idx_of_match:]
            if idx != 0:
                concatenated_str += ' '.join(row_data[-idx:])
            return concatenated_str
    return row['Code/Description']



def remove_before_total(df):
    for index, row in df.iterrows():
        if "TOTAL" in str(row['Code/Description']):
            row_data = str(row['Code/Description']).split("TOTAL")
            df.at[index, 'Code/Description'] = "TOTAL" + row_data[-1]
    return df



def is_numeric(value):
    return bool(re.search(r'^\d+\.\d+$', str(value)))

def clean_numeric_columns(row):
    if not is_numeric(row[PYA]):
        row[PYA] = None
    if not is_numeric(row[CYA]):
        row[CYA] = None
    return row

def merge_rows(df):
    i = 0
    while i < len(df) - 1:
        if not pd.isna(df.loc[i, 'Code']) and pd.isna(df.loc[i, CYA]) and pd.isna(df.loc[i, PYA]):
            j = i + 1
            while j < len(df) and pd.isna(df.loc[j, 'Code']):
                df.loc[i, 'Code/Description'] += ' ' + df.loc[j, 'Code/Description']
                if not pd.isna(df.loc[j, CYA]) or not pd.isna(df.loc[j, PYA]):
                    df.loc[i, CYA] = df.loc[j, CYA]
                    df.loc[i, PYA] = df.loc[j, PYA]
                    df.drop(list(range(i+1, j+1)), inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    break
                j += 1
            else:
                i = j
        else:
            i += 1
    return df


def concatenate_rows(df):
    pattern = r'(?:[A-Z]-|\([a-z]\)|\([ivx]+\))'  # matches A-, (a), (ii) etc.

    to_drop = []
    for idx, row in df.iterrows():
        if re.search(pattern, str(row['Code/Description'])):
            next_idx = idx + 1
            while next_idx < len(df) and pd.isna(df.at[next_idx, CYA]) and pd.isna(
                    df.at[next_idx, PYA]) and not re.search(pattern, str(df.at[next_idx, 'Code/Description'])):
                df.at[idx, 'Code/Description'] += ' ' + df.at[next_idx, 'Code/Description']
                to_drop.append(next_idx)
                next_idx += 1

    df.drop(to_drop, axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# Directory containing your JSON files
dir_path = 'doc_jsons/20212022'  # Modify this if your json files are in another directory
PYA, CYA = f"{int(dir_path[-8:-4])-1}-{dir_path[-8:-4]}", dir_path[-8:-4] + '-' + dir_path[-4:]
# Output Excel file
output_excel = "output_oci.xlsx"

# Get all JSON files from the directory
json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]

json_files = sorted(json_files, key=extract_number)
print(json_files)
all_dataframes = []

# for json_file in json_files:
#     df = process_json(os.path.join(dir_path, json_file))
#     all_dataframes.append(df)

df = process_json(os.path.join(dir_path, json_files[3]))
all_dataframes.append(df)

# 4. Consolidate all data into a single Excel sheet
df = pd.concat(all_dataframes, ignore_index=True)

df


# In[4]:


#ADDING HEADINGS TO TABLE
new_columns = ["HindiCode", "Code/Description", CYA, PYA]
remaining_columns = df.columns[len(new_columns):]
df.columns = new_columns + list(remaining_columns)
# df.columns = ["HindiCode"] + ["Code/Description"] + [CYA, PYA] + df.columns[4:]
df


# In[5]:


# # Create a mask for rows meeting the condition
mask = (
    df[PYA].isna() &
    ~df[CYA].isna() &
    df['Code/Description'].astype(str).str.replace(r'^\*', '', regex=True).str.replace('.', '', 1).str.isnumeric()
)


df.loc[mask, PYA] = df.loc[mask, CYA]
df.loc[mask, CYA] = df.loc[mask, 'Code/Description']
df.loc[mask, 'Code/Description'] = df.loc[mask, 'HindiCode']
df.loc[mask, 'HindiCode'] = None

df




df['Code/Description'] = df.apply(process_row, axis=1)
df



# In[7]:


# Filter rows where 'HindiCode' is not NaN, contains "total" (case-insensitive), and 'Code/Description' is NaN
condition = (df['HindiCode'].notna() & 
             df['HindiCode'].str.contains('total', case=False, na=False) & 
             df['Code/Description'].isna())

# Copy values from 'HindiCode' to 'Code/Description' for rows matching the condition
df.loc[condition, 'Code/Description'] = df.loc[condition, 'HindiCode']
df


# In[8]:



df[[PYA,CYA]] = df[[PYA,CYA]].applymap(lambda x: float(re.sub('[^0-9.]', '', str(x))) if re.search(r'[*]?\d+\.\d+', str(x)) else x)
df


# In[9]:


df.to_csv('raw.csv')


# In[10]:




# Apply the function to the DataFrame
df = df.apply(shift_values, axis=1)

df


# In[11]:


df.to_csv('raw1.csv')


# In[12]:



df['Code/Description'] = df.apply(process_row, axis=1)
df




# In[13]:



# Sample usage:
df = remove_before_total(df)
df


# In[14]:


# Check for the presence of the word 'Disbursement' and 'Receipts'
contains_disbursement = df['HindiCode'].str.contains('Disbursements').any()
print(contains_disbursement)
contains_receipts = df['HindiCode'].str.contains('Receipts').any()
print(contains_receipts)


# In[15]:


df = df.drop(df.columns[0], axis=1)
df


# In[16]:


i = 0
while i < len(df):
    row = df.iloc[i]
    if 'TOTAL' in str(row['Code/Description']).upper() and pd.isna(row[PYA]) and pd.isna(row[CYA]):
        # Start of a section to be concatenated
        combined_desc = row['Code/Description']
        
        j = i + 1  # Initialize the next row index
        while j < len(df):
            next_row = df.iloc[j]
            is_numeric_2020_2021 = pd.notna(pd.to_numeric(next_row[PYA], errors='coerce'))
            is_numeric_2021_2022 = pd.notna(pd.to_numeric(next_row[CYA], errors='coerce'))
            
            combined_desc += " " + next_row['Code/Description']
            if is_numeric_2020_2021 or is_numeric_2021_2022:
                # Update the current row's description and numeric columns in-place
                df.loc[i, 'Code/Description'] = combined_desc
                df.loc[i, PYA] = next_row[PYA]
                df.loc[i, CYA] = next_row[CYA]
                break
                
            j += 1

        # Drop rows from i+1 to j (both inclusive) as they've been combined into the row at index i
        df.drop(range(i+1, j+1), inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    else:
        i += 1


# In[17]:


df = df.dropna(how='all')
df


# In[18]:


df['Code'] = df['Code/Description'].str.extract(r'(\b\d{4}\b)')
# Replace 4-digit code followed by non-alphabetical characters up to the first alphabetical character
df['Code/Description'] = df['Code/Description'].str.replace(r'\b\d{4}\b[^\w]*', '', regex=True)
df


# In[19]:


# Set the column order to make 'Code' the first column
df = df[['Code'] + [col for col in df if col != 'Code']]
df


# In[20]:


df.reset_index(drop=True, inplace=True)
df


# In[21]:


# Remove rows where any column contains CYA or PYA
mask = df.apply(lambda row: (CYA not in row.values) and (PYA not in row.values), axis=1)
df = df[mask]
df.reset_index(drop=True, inplace=True)
df


# In[22]:


df = df.apply(clean_numeric_columns, axis=1)
df


# In[23]:




df = merge_rows(df)
df


# In[24]:


df.reset_index(drop=True, inplace=True)
df


# In[25]:


df.reset_index(drop=True, inplace=True)




df = concatenate_rows(df)
df


# In[26]:


rows_to_drop = []

# Loop through the dataframe but exclude the last row for comparison purposes
for idx in range(len(df) - 1):
    current_row = df.iloc[idx]
    next_row = df.iloc[idx + 1]

    # Condition to check if the current row and the next row meet the criteria
    if pd.notna(current_row['Code']) and pd.isna(current_row[PYA]) and pd.isna(current_row[CYA]) \
            and pd.isna(next_row['Code']) and pd.notna(next_row[PYA]) and pd.notna(next_row[CYA]):
        # Merge data from the next row into the current row
        df.at[idx, PYA] = next_row[PYA]
        df.at[idx, CYA] = next_row[CYA]

        # Mark the next row for deletion
        rows_to_drop.append(idx + 1)

# Drop the rows marked for deletion
df.drop(rows_to_drop, inplace=True)
df.reset_index(drop=True, inplace=True)
df


# 

# In[27]:


mask = (df['Code/Description'] == '1') & (df[PYA] == '3') & (df[CYA] == '2')

# Use the inverse of the mask to filter out rows that match the criteria
df = df[~mask]
df


# In[28]:


# Initializing columns
df['header'] = None
df['subheader'] = None
df['sub-subheader'] = None

# Placeholder variables for each header type
current_header = None
current_subheader = None
current_subsubheader = None

for idx, row in df.iterrows():
    desc = row['Code/Description']
    
    # Check for each pattern and update the corresponding placeholders
    if re.search(r"^[A-Z]-", desc):
        current_header = re.split(r"^[A-Z]-", desc)[-1].strip()
        current_subheader = None
        current_subsubheader = None
        
    elif re.search(r"\([a-z]\)", desc):
        current_subheader = re.split(r"\([a-z]\)", desc)[-1].strip()
        current_subsubheader = None
        
    elif re.search(r"\([ivx]+\)", desc):
        current_subsubheader = re.split(r"\([ivx]+\)", desc)[-1].strip()
        
    # Assigning the values to the new columns
    df.at[idx, 'header'] = current_header if current_header else desc
    df.at[idx, 'subheader'] = current_subheader if current_subheader else desc
    df.at[idx, 'sub-subheader'] = current_subsubheader if current_subsubheader else desc

df



# In[29]:


df = df.dropna(subset=['Code', PYA, CYA], how='all').reset_index(drop=True)
df


# In[30]:


df = df.dropna(subset=[PYA, CYA], how='all')
df


# In[31]:


df = df.dropna(how='all')

df


# In[32]:


df = df.reset_index(drop=True)
df


# In[33]:


# Define a condition to check for alphabetic characters
condition = (df[PYA].astype(str).str.contains('[a-zA-Z]', na=False)) | \
            (df[CYA].astype(str).str.contains('[a-zA-Z]', na=False))

# Remove rows that match the condition
df = df[~condition].reset_index(drop=True)
df


# In[34]:


# Melt the DataFrame to change its structure
melted_df = df.melt(id_vars=['Code', 'Code/Description', 'header', 'subheader', 'sub-subheader'], 
                    value_vars=[CYA, PYA],
                    var_name='Year',
                    value_name='Amount')

# Extract the actual year from the 'Year' column
melted_df['Year'] = melted_df['Year'].str.split('-').str[1].astype(int)
melted_df.reset_index(drop=True, inplace=True)
# Sort the DataFrame based on 'Code' and 'Year'
# melted_df = melted_df.sort_values(by=['Year'], ascending=[False]).reset_index(drop=True)

melted_df


# In[35]:


# Add the column based on conditions
if contains_disbursement:
    melted_df['page_type'] = 'Disbursements'
elif contains_receipts:
    melted_df['page_type'] = 'Receipts'
else:
    melted_df['page_type'] = None

melted_df


# In[36]:


melted_df.to_csv('./output_format1_20212022_test.csv')

