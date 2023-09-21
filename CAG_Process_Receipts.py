#!/usr/bin/env python

import json
import pandas as pd
import os
import re
from difflib import get_close_matches
from fuzzywuzzy import fuzz

HEADERS = [
    "NON-TAXREVENUE",
    "TAXREVENUE",
    "GENERAL SERVICES",
    "SOCIAL SERVICES",
    "GRANTS-IN-AID AND CONTRIBUTIONS",
    "CAPITAL RECEIPT",
    "PUBLIC DEBT",
    "LOANS AND ADVANCES",
    "ECONOMIC SERVICES",
    "INTERSTATE SETTLEMENT",
    "Transfer to Contigency Fund",
    "SMALL SAVINGS,PROVIDENT FUND",
    "RESERVE FUNDS",
    "DEPOSITS AND ADVANCES",
    "SUSPENSE AND MISCELLANEOUS",
    "REMITTANCES",
    "TOTAL-PUBLIC ACCOUNT",
    "TOTAL-RECEIPTS",
    "OPENING CASH BALANCE",
    "GRAND TOTAL"
]

SUBHEADERS = [
    "Goods and Service Tax",
    "Taxes on Income and Expenditure",
    "Organs of State",
    "Fiscal Services",
    "Taxes on Property, Capital and other Transactions",
    "Taxes on Commodities and Services Other than Goods and Services Tax",
    "Interest Payments and Servicing of Debt",
    "Interests Receipts, Dividents and Profits",
    "Other Non-Tax Revenue",
    "Administrative Services",
    "Pension and Miscellaneous General Services",
    "Defense Services",
    "Education, Sports, Art and Culture",
    "Health and Family Welfare",
    "Water Supply, Sanitation, Housing and Urban Development",
    "Information and Broadcasting",
    "Welfare of Schedules Castes, Scheduled Tribes and Other Backward Classes",
    "Labour and Labour Welfare",
    "Social Welfare and Nutrition",
    "Others",
    "Agriculture and Allied Activities",
    "Rural Development",
    "Special Areas Programmes",
    "Irrigation and Flood Control",
    "Energy",
    "Industry and Minerals",
    "Transport",
    "Communications",
    "Science Technology and Environment",
    "General Economic Services",
    "Reserve Funds bearing interest",
    "Reserve Funds not bearing interest",
    "Deposits bearing interest",
    "Deposits not bearing interest",
    "Advances"
]

SUBSUBHEADERS = [
    "Collection of Taxes on Income and Expenditure",
    "Collection of Taxes on Property and Capital Transactions",
    "Collection of Taxes on Commodities and Services",
    "Other Fiscal Services",
    "General Services",
    "Social Services",
    "Economic Services"
]
def extract_amount(s):
    match = re.search(r'(\d+\.\d+)', s)
    return match.group(1) if match else None

def remove_amount(s):
    return re.sub(r'\d+\.\d+', '', s).strip()

def clean_string(s):
    # Remove numbers, special characters, and extra whitespace
    s = re.sub(r'[^a-zA-Z\s]', '', s)
    return s.strip()

def get_closest_match(input_str, possible_matches):
    direct_matches = [match for match in possible_matches if match.lower() in input_str.lower()]
    # If there's more than one direct match, return the longest one
    if direct_matches:
        return max(direct_matches, key=len)
    """Get the closest match for the input string from a list of possible matches."""
    matches = get_close_matches(input_str, possible_matches, n=1, cutoff=0.6)
    return matches[0] if matches else input_str

def get_best_match(input_str, possible_matches):
    
    direct_matches = [match for match in possible_matches if match.lower() in input_str.lower()]    
    if direct_matches:
        return max(direct_matches, key=len)
    ratios = [(match, fuzz.token_set_ratio(input_str, match)) for match in possible_matches]
    best_match = max(ratios, key=lambda x: x[1])
    if best_match[1] > 60:  # Assuming 60 as a threshold. Adjust if necessary.
        return best_match[0]
    return input_str

def process_headers(df):
    # Initialize the columns with NaN values
    df['header'] = pd.NA
    df['subheader'] = pd.NA
    df['sub-subheader'] = pd.NA
    
    # Placeholder variables for each header type
    current_header = None
    current_subheader = None
    current_subsubheader = None
    
    # Flag to handle the situation where a subheader is followed by a Roman numeral
    expecting_subsubheader = False

    for idx, row in df.iterrows():
        desc = row['Code/Description']

        # Ensure code is NaN
        if pd.isna(row['Code']):
            
            # If the last row was a subheader and this row is a Roman numeral, treat it as a sub-subheader
            if expecting_subsubheader:
                subsubheader_match = re.search(r"\((i|ii|iii|iv|v|vi|vii|viii|ix|x)\)", desc)
                if subsubheader_match:
                    current_subsubheader = get_closest_match(clean_string(desc.strip()), SUBSUBHEADERS)
                    continue
                else:
                    expecting_subsubheader = False
            
            # Check for header pattern
            header_match = re.search(r"^(?:[A-Z]-|[A-Z]+$)", desc)
            print(header_match)
            if header_match:
                print(header_match)
                current_header = get_closest_match(desc[header_match.end():].strip(), HEADERS)
                current_subheader = None
                current_subsubheader = None

            # Check for subheader pattern
            subheader_match = re.search(r"\([a-z]\)", desc)
            if subheader_match:
                current_subheader = get_closest_match((desc[subheader_match.end():].strip()), SUBHEADERS)
                current_subsubheader = None
                # If a subheader is detected, set the flag
                expecting_subsubheader = True

            # If a sub-subheader pattern is detected without the flag being set, just handle it normally
            if not expecting_subsubheader:
                subsubheader_match = re.search(r"\((i|ii|iii|iv|v|vi|vii|viii|ix|x)\)", desc)
                if subsubheader_match:
                    current_subsubheader = get_closest_match((desc[subsubheader_match.end():].strip()), SUBSUBHEADERS)
        
        # Update DataFrame rows with the current header values
        df.at[idx, 'header'] = current_header
        df.at[idx, 'subheader'] = current_subheader
        df.at[idx, 'sub-subheader'] = current_subsubheader
    
    return df



def check_values_in_dataframe(df):
    # Define regular expressions for the target values
    exp_notes_pattern = r'.*E\s*X\s*P\s*L\s*A\s*N\s*A\s*T\s*O\s*R\s*Y\s*N\s*O\s*T\s*E\s*S.*'
    actuals_pattern = r'.*A\s*C\s*T\s*U\s*A\s*L\s*S.*'

    # Iterate through all the values in the DataFrame
    for column in df.columns:
        for index, value in df[column].items():
            if value is None:
                continue
            # Remove spaces and convert to lowercase for comparison
            cleaned_value = re.sub(r'\s', '', value.upper())

            # Check for "EXPLANATORYNOTES" using regular expression
            if re.match(exp_notes_pattern, cleaned_value):
                return "exp_notes"

            # Check for "Actuals" using regular expression
            if re.match(actuals_pattern, cleaned_value):
                return "receipts"

    # If none of the conditions match, return "others"
    return "others"

# Function to check entire row for keywords
def check_keywords(row):
    if 'Receipts' in row.to_string():
        return 'R'
    elif 'Disbursements' in row.to_string():
        return 'D'
    return None


def extract_number(filename):
    match = re.search(r'output_pg_(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return 0  # default value if no match is found


# In[3]:


def has_decimal_number(item):
    return bool(re.search(r'^\d+\.\d+$', item))


def process_row(row, PYA, CYA):
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


# In[4]:


# find heading for the total and add it into a separate column for all the rows contained in it

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
# pattern = r'(?:[A-Z]-|\([a-z]\)|\([ivx]+\))'
pattern = r'(?:^[A-M]-|\([a-l]\)|\([ivx]+\))'
def correct_ocr_errors(s):
    # If the string starts with an uppercase letter followed by an uppercase letter without a hyphen, 
    # we'll insert a hyphen.
    # corrected = re.sub(r'^([A-Z])([A-Z])', r'\1-\2', s)
    return s

def has_numeric(row, PYA, CYA):
    return bool(re.search(r'^\d+(\.\d+)?$', str(row[PYA]))) and bool(re.search(r'^\d+(\.\d+)?$', str(row[CYA])))


def process_row1(row, PYA, CYA):
    # If there are numeric values in the row, just return the current value
    if has_numeric(row, PYA, CYA) or ("TOTAL" in str(row['Code/Description'])):
        return row['Code/Description']

    row_data = [str(item) for item in row if pd.notnull(item)]
    for idx, item in enumerate(reversed(row_data)):
        corrected_item = correct_ocr_errors(item)
        match = re.search(pattern, corrected_item)
        if match:
            idx_of_match = corrected_item.rfind(match.group())
            concatenated_str = corrected_item[idx_of_match:]
            if idx != 0:
                concatenated_str += ' '.join(row_data[-idx:])
            return concatenated_str
    return row['Code/Description']

# In[6]:


def remove_before_total(df):
    for index, row in df.iterrows():
        if "TOTAL" in str(row['Code/Description']):
            row_data = str(row['Code/Description']).split("TOTAL")
            df.at[index, 'Code/Description'] = "TOTAL" + row_data[-1]
    return df


# In[7]:


def is_numeric(value):
    return bool(re.search(r'^\d+\.\d+$', str(value)))


def clean_numeric_columns(row, PYA, CYA):
    if not is_numeric(row[PYA]):
        row[PYA] = None
    if not is_numeric(row[CYA]):
        row[CYA] = None
    return row


# In[8]:


def merge_rows(df, PYA, CYA):
    i = 0
    while i < len(df) - 1:
        if not pd.isna(df.loc[i, 'Code']) and pd.isna(df.loc[i, CYA]) and pd.isna(df.loc[i, PYA]):
            j = i + 1
            while j < len(df) and pd.isna(df.loc[j, 'Code']):
                df.loc[i, 'Code/Description'] += ' ' + df.loc[j, 'Code/Description']
                if not pd.isna(df.loc[j, CYA]) or not pd.isna(df.loc[j, PYA]):
                    df.loc[i, CYA] = df.loc[j, CYA]
                    df.loc[i, PYA] = df.loc[j, PYA]
                    df.drop(list(range(i + 1, j + 1)), inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    break
                j += 1
            else:
                i = j
        else:
            i += 1
    return df


# In[9]:


def concatenate_rows(df, PYA, CYA):
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


# Define the function to process one JSON file
def process_json(data, PYA, CYA):

    ocr_data = {
        'lines': data['image_text']['lines']
        # 'lines': data['image_text']['lines']
    }

    # Thresholds
    y_threshold = 0.01
    x_threshold = 0.03

    rows = []

    for line in ocr_data["lines"]:
        y_avg = sum([v["y"] for v in line["bounding_polygon"]["normalized_vertices"]]) / 4
        x_start = line["bounding_polygon"]["normalized_vertices"][0]["x"]
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
    format_type = check_values_in_dataframe(df)
    if format_type == 'others' or format_type == 'exp_notes':
        return format_type
    df = dataframe_processing(df, PYA, CYA)

    return df



def dataframe_processing(df, PYA, CYA):
    # ADDING HEADINGS TO TABLE
    new_columns = ["HindiCode", "Code/Description", CYA, PYA]
    remaining_columns = df.columns[len(new_columns):]
    df.columns = new_columns + list(remaining_columns)
    # Create a mask for rows meeting the condition
    mask = (
            df[PYA].isna() &
            ~df[CYA].isna() &
            df['Code/Description'].astype(str).str.replace(r'^\*', '', regex=True).str.replace('.', '',
                                                                                               1).str.isnumeric()
    )

    df.loc[mask, PYA] = df.loc[mask, CYA]
    df.loc[mask, CYA] = df.loc[mask, 'Code/Description']
    df.loc[mask, 'Code/Description'] = df.loc[mask, 'HindiCode']
    df.loc[mask, 'HindiCode'] = None

    contains_disbursement = False
    pattern_disbursements = r'D\s*i\s*s\s*b\s*u\s*r\s*s\s*e\s*m\s*e\s*n\s*t\s*s'
    for col in df.columns:
        if df[col].astype(str).str.contains(pattern_disbursements, regex=True).any():
            contains_disbursement = True
            break

    contains_receipts = False
    pattern_receipts = r'R\s*e\s*c\s*e\s*i\s*p\s*t\s*s'
    for col in df.columns:
        if df[col].astype(str).str.contains(pattern_receipts, regex=True).any():
            contains_receipts = True
            break

    # In[13]:

    df['Code/Description'] = df.apply(process_row, axis=1, args=(PYA, CYA))

    # In[14]:
    
    # Filter rows where 'HindiCode' is not NaN, contains "total" (case-insensitive), and 'Code/Description' is NaN
    condition = (df['HindiCode'].notna() &
                 df['HindiCode'].str.contains('total', case=False, na=False) &
                 df['Code/Description'].isna())

    # Copy values from 'HindiCode' to 'Code/Description' for rows matching the condition
    df.loc[condition, 'Code/Description'] = df.loc[condition, 'HindiCode']
    
#     # Identify the condition
#     condition = (df['Code/Description'].astype(str).str.replace(".", "", 1).str.isnumeric()) & (df['HindiCode'].notna())

#     # Copy values based on the condition
#     df.loc[condition, PYA] = df.loc[condition, 'Code/Description']
#     df.loc[condition, 'Code/Description'] = df.loc[condition, 'HindiCode']
#     df.loc[condition, CYA] = df.loc[condition, 'Code/Description'].apply(extract_amount)
#     df.loc[condition, 'Code/Description'] = df.loc[condition, 'Code/Description'].apply(remove_amount)
    
    df[[PYA, CYA]] = df[[PYA, CYA]].applymap(
        lambda x: float(re.sub('[^0-9.]', '', str(x))) if re.search(r'[*]?\d+\.\d+', str(x)) else x)
    
    df = df.apply(shift_values, axis=1)
    
    df['Code/Description'] = df.apply(process_row1, axis=1, args=(PYA, CYA))
    
    # Sample usage:
    df = remove_before_total(df)
    
    df = df.drop(df.columns[0], axis=1)

    # In[23]:
    
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
            df.drop(range(i + 1, j + 1), inplace=True)
            df.reset_index(drop=True, inplace=True)

        else:
            i += 1

    # In[24]:
    df = df.dropna(how='all')
    
    # In[25]:

    df['Code'] = df['Code/Description'].str.extract(r'(\b\d{4}\b)')
    # Replace 4-digit code followed by non-alphabetical characters up to the first alphabetical character
    df['Code/Description'] = df['Code/Description'].str.replace(r'\b\d{4}\b[^\w]*', '', regex=True)
    
    # In[26]:

    # Set the column order to make 'Code' the first column
    df = df[['Code'] + [col for col in df if col != 'Code']]

    # In[27]:

    df.reset_index(drop=True, inplace=True)
    
    # In[28]:
    
    # Remove rows where any column contains CYA or PYA
    mask = df.apply(lambda row: (CYA not in row.values) and (PYA not in row.values), axis=1)
    df = df[mask]
    df.reset_index(drop=True, inplace=True)

    df = df.apply(clean_numeric_columns, axis=1, args=(PYA, CYA))

    df = merge_rows(df, PYA, CYA)
    
    # In[31]:

    df.reset_index(drop=True, inplace=True)
    
    # In[32]:

    df.reset_index(drop=True, inplace=True)

    df = concatenate_rows(df, PYA, CYA)

    # In[33]:

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

    mask = (df['Code/Description'] == '1') & (df[PYA] == '3') & (df[CYA] == '2')

    # Use the inverse of the mask to filter out rows that match the criteria
    df = df[~mask]
    
#     # Initializing columns
#     df['header'] = pd.NA
#     df['subheader'] = pd.NA
#     df['sub-subheader'] = pd.NA

#     # Placeholder variables for each header type
#     current_header = None
#     current_subheader = None
#     current_subsubheader = None

#     for idx, row in df.iterrows():
#         desc = row['Code/Description']

#         # Check for each pattern and update the corresponding placeholders
#         if re.search(r"^[A-Z]-", desc):
#             current_header = re.split(r"^[A-Z]-", desc)[-1].strip()
#             current_subheader = None
#             current_subsubheader = None

#         elif re.search(r"\([a-z]\)", desc):
#             current_subheader = re.split(r"\([a-z]\)", desc)[-1].strip()
#             current_subsubheader = None

#         elif re.search(r"\([ivx]+\)", desc):
#             current_subsubheader = re.split(r"\([ivx]+\)", desc)[-1].strip()

#         # Assigning the values to the new columns
#         df.at[idx, 'header'] = current_header if current_header else desc
#         df.at[idx, 'subheader'] = current_subheader if current_subheader else desc
#         df.at[idx, 'sub-subheader'] = current_subsubheader if current_subsubheader else desc

    # df[['header', 'subheader', 'sub-subheader']] = df['Code/Description'].apply(process_headers)
    print(df)
    df = process_headers(df)

    df = df.dropna(subset=['Code', PYA, CYA], how='all').reset_index(drop=True)
    
    df = df.dropna(subset=[PYA, CYA], how='all')

    df = df.dropna(how='all')
    
    df = df.reset_index(drop=True)
    
    # Define a condition to check for alphabetic characters
    condition = (df[PYA].astype(str).str.contains('[a-zA-Z]', na=False)) & \
                (df[CYA].astype(str).str.contains('[a-zA-Z]', na=False))

    # Remove rows that match the condition
    df = df[~condition].reset_index(drop=True)
    
    # In[41]:
    df['Code'] = df['Code'].astype(str)

    # Melt the DataFrame to change its structure
    melted_df = df.melt(id_vars=['Code', 'Code/Description', 'header', 'subheader', 'sub-subheader'],
                        value_vars=[CYA, PYA],
                        var_name='Year',
                        value_name='Amount')
    
    # Extract the actual year from the 'Year' column
    melted_df['Year'] = melted_df['Year'].str.split('-').str[1].astype(int)
    melted_df.reset_index(drop=True, inplace=True)

    # Add the column based on conditions
    if contains_disbursement:
        melted_df['page_type'] = 'Disbursements'
    elif contains_receipts:
        melted_df['page_type'] = 'Receipts'
    else:
        melted_df['page_type'] = None
    
    
    return melted_df

# #!/usr/bin/env python

# import json
# import pandas as pd
# import os
# import re


# def check_values_in_dataframe(df):
#     # Define regular expressions for the target values
#     exp_notes_pattern = r'.*E\s*X\s*P\s*L\s*A\s*N\s*A\s*T\s*O\s*R\s*Y\s*N\s*O\s*T\s*E\s*S.*'
#     actuals_pattern = r'.*A\s*C\s*T\s*U\s*A\s*L\s*S.*'

#     # Iterate through all the values in the DataFrame
#     for column in df.columns:
#         for index, value in df[column].items():
#             if value is None:
#                 continue
#             # Remove spaces and convert to lowercase for comparison
#             cleaned_value = re.sub(r'\s', '', value.upper())

#             # Check for "EXPLANATORYNOTES" using regular expression
#             if re.match(exp_notes_pattern, cleaned_value):
#                 return "exp_notes"

#             # Check for "Actuals" using regular expression
#             if re.match(actuals_pattern, cleaned_value):
#                 return "receipts"

#     # If none of the conditions match, return "others"
#     return "others"

# # Function to check entire row for keywords
# def check_keywords(row):
#     if 'Receipts' in row.to_string():
#         return 'R'
#     elif 'Disbursements' in row.to_string():
#         return 'D'
#     return None


# def extract_number(filename):
#     match = re.search(r'output_pg_(\d+)', filename)
#     if match:
#         return int(match.group(1))
#     else:
#         return 0  # default value if no match is found


# # In[3]:


# def has_decimal_number(item):
#     return bool(re.search(r'^\d+\.\d+$', item))


# def process_row(row, PYA, CYA):
#     # If there are decimal numeric values in the row, just return the current value
#     if has_decimal_number(str(row[PYA])) or has_decimal_number(str(row[CYA])):
#         return row['Code/Description']

#     cols = row.index
#     row_data = [str(item) for item in row if pd.notnull(item)]
#     for idx, item in enumerate(reversed(row_data)):
#         match = re.search(r'\b\d{4}\b', item)
#         if match:
#             idx_of_match = item.rfind(match.group())
#             concatenated_str = item[idx_of_match:]
#             if idx != 0:
#                 concatenated_str += ' ' + ' '.join(row_data[-idx:])
#             return concatenated_str
#     return row['Code/Description']


# # In[4]:


# # find heading for the total and add it into a separate column for all the rows contained in it

# def shift_values(row):
#     # Check if Code/Description is None
#     if pd.isna(row['Code/Description']):
#         # Search for the pattern of a 4-digit code from the end of the string
#         match = re.search(r'(\d{4}-.*$)', row['HindiCode'])
#         if match:
#             # Assign everything including and after the 4-digit code to Code/Description
#             row['Code/Description'] = match.group(1)
#             # Extract everything before the 4-digit code in HindiCode
#             row['HindiCode'] = row['HindiCode'].rsplit(match.group(1), 1)[0].strip()
#     return row


# # Pattern to detect headers, subheaders, and sub-subheaders
# pattern = r'(?:[A-Z]-|\([a-z]\)|\([ivx]+\))'


# def has_numeric(row, PYA, CYA):
#     return bool(re.search(r'^\d+(\.\d+)?$', str(row[PYA]))) and bool(re.search(r'^\d+(\.\d+)?$', str(row[CYA])))


# def process_row1(row, PYA, CYA):
#     # If there are numeric values in the row, just return the current value
#     if has_numeric(row, PYA, CYA) or ("TOTAL" in str(row['Code/Description'])):
#         return row['Code/Description']

#     row_data = [str(item) for item in row if pd.notnull(item)]
#     for idx, item in enumerate(reversed(row_data)):
#         match = re.search(pattern, item)
#         if match:
#             idx_of_match = item.rfind(match.group())
#             concatenated_str = item[idx_of_match:]
#             if idx != 0:
#                 concatenated_str += ' '.join(row_data[-idx:])
#             return concatenated_str
#     return row['Code/Description']


# # In[6]:


# def remove_before_total(df):
#     for index, row in df.iterrows():
#         if "TOTAL" in str(row['Code/Description']):
#             row_data = str(row['Code/Description']).split("TOTAL")
#             df.at[index, 'Code/Description'] = "TOTAL" + row_data[-1]
#     return df


# # In[7]:


# def is_numeric(value):
#     return bool(re.search(r'^\d+\.\d+$', str(value)))


# def clean_numeric_columns(row, PYA, CYA):
#     if not is_numeric(row[PYA]):
#         row[PYA] = None
#     if not is_numeric(row[CYA]):
#         row[CYA] = None
#     return row


# # In[8]:


# def merge_rows(df, PYA, CYA):
#     i = 0
#     while i < len(df) - 1:
#         if not pd.isna(df.loc[i, 'Code']) and pd.isna(df.loc[i, CYA]) and pd.isna(df.loc[i, PYA]):
#             j = i + 1
#             while j < len(df) and pd.isna(df.loc[j, 'Code']):
#                 df.loc[i, 'Code/Description'] += ' ' + df.loc[j, 'Code/Description']
#                 if not pd.isna(df.loc[j, CYA]) or not pd.isna(df.loc[j, PYA]):
#                     df.loc[i, CYA] = df.loc[j, CYA]
#                     df.loc[i, PYA] = df.loc[j, PYA]
#                     df.drop(list(range(i + 1, j + 1)), inplace=True)
#                     df.reset_index(drop=True, inplace=True)
#                     break
#                 j += 1
#             else:
#                 i = j
#         else:
#             i += 1
#     return df


# # In[9]:


# def concatenate_rows(df, PYA, CYA):
#     pattern = r'(?:[A-Z]-|\([a-z]\)|\([ivx]+\))'  # matches A-, (a), (ii) etc.

#     to_drop = []
#     for idx, row in df.iterrows():
#         if re.search(pattern, str(row['Code/Description'])):
#             next_idx = idx + 1
#             while next_idx < len(df) and pd.isna(df.at[next_idx, CYA]) and pd.isna(
#                     df.at[next_idx, PYA]) and not re.search(pattern, str(df.at[next_idx, 'Code/Description'])):
#                 df.at[idx, 'Code/Description'] += ' ' + df.at[next_idx, 'Code/Description']
#                 to_drop.append(next_idx)
#                 next_idx += 1

#     df.drop(to_drop, axis=0, inplace=True)
#     df.reset_index(drop=True, inplace=True)

#     return df


# # Define the function to process one JSON file
# def process_json(data, PYA, CYA):

#     ocr_data = {
#         'lines': data['image_text']['lines']
#     }

#     # Thresholds
#     y_threshold = 0.01
#     x_threshold = 0.03

#     rows = []

#     for line in ocr_data["lines"]:
#         y_avg = sum([v["y"] for v in line["bounding_polygon"]["normalized_vertices"]]) / 4
#         x_start = line["bounding_polygon"]["normalized_vertices"][0]["x"]
#         added = False
#         for row in rows:
#             if abs(row["y_avg"] - y_avg) < y_threshold:
#                 added_to_cell = False
#                 for cell in row["cells"]:
#                     if abs(cell["x_start"] - x_start) < x_threshold:
#                         cell["texts"].append(line["text"])
#                         added_to_cell = True
#                         break
#                 if not added_to_cell:
#                     row["cells"].append({"x_start": x_start, "texts": [line["text"]]})
#                 added = True
#                 break
#         if not added:
#             rows.append({"y_avg": y_avg, "cells": [{"x_start": x_start, "texts": [line["text"]]}]})

#     # Convert rows to pandas DataFrame
#     data = []
#     for row in rows:
#         out_row = []
#         for cell in row["cells"]:
#             out_row.append(" ".join(cell["texts"]))
#         data.append(out_row)

#     df = pd.DataFrame(data)
#     format_type = check_values_in_dataframe(df)
#     if format_type == 'others' or format_type == 'exp_notes':
#         return format_type
#     df = dataframe_processing(df, PYA, CYA)

#     return df


# # Directory containing your JSON files
# # dir_path = './doc_jsons/20212022'  # Modify this if your json files are in another directory
# # PYA, CYA = f"{int(dir_path[-8:-4]) - 1}-{dir_path[-8:-4]}", dir_path[-8:-4] + '-' + dir_path[-4:]
# # Output Excel file
# # output_excel = "output_oci.xlsx"


# def dataframe_processing(df, PYA, CYA):
#     # ADDING HEADINGS TO TABLE
#     new_columns = ["HindiCode", "Code/Description", CYA, PYA]
#     remaining_columns = df.columns[len(new_columns):]
#     df.columns = new_columns + list(remaining_columns)

#     # Create a mask for rows meeting the condition
#     mask = (
#             df[PYA].isna() &
#             ~df[CYA].isna() &
#             df['Code/Description'].astype(str).str.replace(r'^\*', '', regex=True).str.replace('.', '',
#                                                                                                1).str.isnumeric()
#     )

#     df.loc[mask, PYA] = df.loc[mask, CYA]
#     df.loc[mask, CYA] = df.loc[mask, 'Code/Description']
#     df.loc[mask, 'Code/Description'] = df.loc[mask, 'HindiCode']
#     df.loc[mask, 'HindiCode'] = None

#     contains_disbursement = False
#     pattern_disbursements = r'D\s*i\s*s\s*b\s*u\s*r\s*s\s*e\s*m\s*e\s*n\s*t\s*s'
#     for col in df.columns:
#         if df[col].astype(str).str.contains(pattern_disbursements, regex=True).any():
#             contains_disbursement = True
#             break

#     contains_receipts = False
#     pattern_receipts = r'R\s*e\s*c\s*e\s*i\s*p\s*t\s*s'
#     for col in df.columns:
#         if df[col].astype(str).str.contains(pattern_receipts, regex=True).any():
#             contains_receipts = True
#             break

#     # In[13]:

#     df['Code/Description'] = df.apply(process_row, axis=1, args=(PYA, CYA))

#     # In[14]:

#     # Filter rows where 'HindiCode' is not NaN, contains "total" (case-insensitive), and 'Code/Description' is NaN
#     condition = (df['HindiCode'].notna() &
#                  df['HindiCode'].str.contains('total', case=False, na=False) &
#                  df['Code/Description'].isna())

#     # Copy values from 'HindiCode' to 'Code/Description' for rows matching the condition
#     df.loc[condition, 'Code/Description'] = df.loc[condition, 'HindiCode']

#     df[[PYA, CYA]] = df[[PYA, CYA]].applymap(
#         lambda x: float(re.sub('[^0-9.]', '', str(x))) if re.search(r'[*]?\d+\.\d+', str(x)) else x)

#     df = df.apply(shift_values, axis=1)

#     df['Code/Description'] = df.apply(process_row1, axis=1, args=(PYA, CYA))

#     # Sample usage:
#     df = remove_before_total(df)

#     df = df.drop(df.columns[0], axis=1)

#     # In[23]:

#     i = 0
#     while i < len(df):
#         row = df.iloc[i]
#         if 'TOTAL' in str(row['Code/Description']).upper() and pd.isna(row[PYA]) and pd.isna(row[CYA]):
#             # Start of a section to be concatenated
#             combined_desc = row['Code/Description']

#             j = i + 1  # Initialize the next row index
#             while j < len(df):
#                 next_row = df.iloc[j]
#                 is_numeric_2020_2021 = pd.notna(pd.to_numeric(next_row[PYA], errors='coerce'))
#                 is_numeric_2021_2022 = pd.notna(pd.to_numeric(next_row[CYA], errors='coerce'))

#                 combined_desc += " " + next_row['Code/Description']
#                 if is_numeric_2020_2021 or is_numeric_2021_2022:
#                     # Update the current row's description and numeric columns in-place
#                     df.loc[i, 'Code/Description'] = combined_desc
#                     df.loc[i, PYA] = next_row[PYA]
#                     df.loc[i, CYA] = next_row[CYA]
#                     break

#                 j += 1

#             # Drop rows from i+1 to j (both inclusive) as they've been combined into the row at index i
#             df.drop(range(i + 1, j + 1), inplace=True)
#             df.reset_index(drop=True, inplace=True)

#         else:
#             i += 1

#     # In[24]:

#     df = df.dropna(how='all')

#     # In[25]:

#     df['Code'] = df['Code/Description'].str.extract(r'(\b\d{4}\b)')
#     # Replace 4-digit code followed by non-alphabetical characters up to the first alphabetical character
#     df['Code/Description'] = df['Code/Description'].str.replace(r'\b\d{4}\b[^\w]*', '', regex=True)

#     # In[26]:

#     # Set the column order to make 'Code' the first column
#     df = df[['Code'] + [col for col in df if col != 'Code']]

#     # In[27]:

#     df.reset_index(drop=True, inplace=True)

#     # In[28]:

#     # Remove rows where any column contains CYA or PYA
#     mask = df.apply(lambda row: (CYA not in row.values) and (PYA not in row.values), axis=1)
#     df = df[mask]
#     df.reset_index(drop=True, inplace=True)

#     df = df.apply(clean_numeric_columns, axis=1, args=(PYA, CYA))

#     df = merge_rows(df, PYA, CYA)

#     # In[31]:

#     df.reset_index(drop=True, inplace=True)

#     # In[32]:

#     df.reset_index(drop=True, inplace=True)

#     df = concatenate_rows(df, PYA, CYA)

#     # In[33]:

#     rows_to_drop = []

#     # Loop through the dataframe but exclude the last row for comparison purposes
#     for idx in range(len(df) - 1):
#         current_row = df.iloc[idx]
#         next_row = df.iloc[idx + 1]

#         # Condition to check if the current row and the next row meet the criteria
#         if pd.notna(current_row['Code']) and pd.isna(current_row[PYA]) and pd.isna(current_row[CYA]) \
#                 and pd.isna(next_row['Code']) and pd.notna(next_row[PYA]) and pd.notna(next_row[CYA]):
#             # Merge data from the next row into the current row
#             df.at[idx, PYA] = next_row[PYA]
#             df.at[idx, CYA] = next_row[CYA]

#             # Mark the next row for deletion
#             rows_to_drop.append(idx + 1)

#     # Drop the rows marked for deletion
#     df.drop(rows_to_drop, inplace=True)
#     df.reset_index(drop=True, inplace=True)

#     mask = (df['Code/Description'] == '1') & (df[PYA] == '3') & (df[CYA] == '2')

#     # Use the inverse of the mask to filter out rows that match the criteria
#     df = df[~mask]

#     # Initializing columns
#     df['header'] = None
#     df['subheader'] = None
#     df['sub-subheader'] = None

#     # Placeholder variables for each header type
#     current_header = None
#     current_subheader = None
#     current_subsubheader = None

#     for idx, row in df.iterrows():
#         desc = row['Code/Description']

#         # Check for each pattern and update the corresponding placeholders
#         if re.search(r"^[A-Z]-", desc):
#             current_header = re.split(r"^[A-Z]-", desc)[-1].strip()
#             current_subheader = None
#             current_subsubheader = None

#         elif re.search(r"\([a-z]\)", desc):
#             current_subheader = re.split(r"\([a-z]\)", desc)[-1].strip()
#             current_subsubheader = None

#         elif re.search(r"\([ivx]+\)", desc):
#             current_subsubheader = re.split(r"\([ivx]+\)", desc)[-1].strip()

#         # Assigning the values to the new columns
#         df.at[idx, 'header'] = current_header if current_header else desc
#         df.at[idx, 'subheader'] = current_subheader if current_subheader else desc
#         df.at[idx, 'sub-subheader'] = current_subsubheader if current_subsubheader else desc

#     df = df.dropna(subset=['Code', PYA, CYA], how='all').reset_index(drop=True)

#     df = df.dropna(subset=[PYA, CYA], how='all')

#     df = df.dropna(how='all')

#     df = df.reset_index(drop=True)

#     # Define a condition to check for alphabetic characters
#     condition = (df[PYA].astype(str).str.contains('[a-zA-Z]', na=False)) | \
#                 (df[CYA].astype(str).str.contains('[a-zA-Z]', na=False))

#     # Remove rows that match the condition
#     df = df[~condition].reset_index(drop=True)

#     # In[41]:
#     df['Code'] = df['Code'].astype(str)

#     # Melt the DataFrame to change its structure
#     melted_df = df.melt(id_vars=['Code', 'Code/Description', 'header', 'subheader', 'sub-subheader'],
#                         value_vars=[CYA, PYA],
#                         var_name='Year',
#                         value_name='Amount')

#     # Extract the actual year from the 'Year' column
#     melted_df['Year'] = melted_df['Year'].str.split('-').str[1].astype(int)
#     melted_df.reset_index(drop=True, inplace=True)

#     # Add the column based on conditions
#     if contains_disbursement:
#         melted_df['page_type'] = 'Disbursements'
#     elif contains_receipts:
#         melted_df['page_type'] = 'Receipts'
#     else:
#         melted_df['page_type'] = None

#     return melted_df


# # # Get all JSON files from the directory
# # json_files = [f for f in os.listdir(dir_path) if f.endswith('.json')]
# #
# # json_files = sorted(json_files, key=extract_number)
# # print(json_files)
# # all_dataframes = []
# #
# # # for json_file in json_files:
# # #     df = process_json(os.path.join(dir_path, json_file))
# # #     p_df = dataframe_processing(df)
# # #     all_dataframes.append(p_df)
# #
# # df = process_json(os.path.join(dir_path, json_files[1]),"2020-2021","2021-2022")
# # # df = dataframe_processing(df)
# # all_dataframes.append(df)
# #
# # # 4. Consolidate all data into a single Excel sheet
# # p_df = pd.concat(all_dataframes, ignore_index=True)
# #
# # p_df.to_csv("./final_format1_20212022.csv")
