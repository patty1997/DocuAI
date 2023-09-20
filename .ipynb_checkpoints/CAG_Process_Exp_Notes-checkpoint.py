import os
import glob
import pandas as pd
import requests
import json
import numpy as np
import re


def has_english_chars(text):
    """Check if the text contains any English character."""
    return bool(re.search('[a-zA-Z]', text))


# function to remove hindi characters
def filter_non_english(s):
    """Remove content that precedes any non-English characters and filter out non-English characters."""
    # Find the position of the first non-ASCII character
    position = next((i for i, c in enumerate(s) if ord(c) >= 128), None)

    # If found, remove everything before it
    if position is not None:
        s = s[position:]

    # Filter out non-ASCII characters
    return ''.join(c for c in s if ord(c) < 128)


# Function to identify the column with the highest number of valid float values
def identify_amount_column(df):
    valid_counts = {}
    is_float_str = lambda x: isinstance(x, str) and x.count('.') == 1 and x.replace('.', '', 1).isdigit()

    for col in df.columns:
        valid_counts[col] = df[col].apply(lambda x: isinstance(x, float) or is_float_str(x)).sum()
    return max(valid_counts, key=valid_counts.get)


def amount_in_char(result):
    # Loop through columns to find the one with the desired pattern
    for col in result.columns:
        mask = result[col].str.contains(r'^[^\d]+\s\d+(\.\d{1,2})?$', na=False)
        if any(mask):
            break

    # If no column matches the desired pattern, print a message and return the original dataframe
    if not any(mask):
        print("No entries in the expected format found!")
        return result

    # Split the identified column
    temp_df = result[mask][col].str.extract(r'(\D+)\s([\d.]+)$')
    temp_df.columns = ['text', 'value']

    # Place the split values into the dataframe
    result.loc[mask, col] = temp_df['text']
    result.insert(col + 1, 'value', temp_df['value'])

    # Drop rows where all values are NaN and reset index
    result = result.dropna(how='all').reset_index(drop=True)

    return result


def handle_missing_values(result):
    # Identify columns by their typical content
    code_col = None
    amount_col = None
    for col in result.columns:
        if result[col].str.contains(r'^\d{4}$', na=False).any():  # Codes typically 4 digits
            code_col = col
        if result[col].str.contains(r'^\d+(\.\d{1,2})?$', na=False).any():  # Amount format
            amount_col = col

    if not code_col or not amount_col:
        print("Couldn't identify necessary columns")
        return result

    rows_to_drop = []
    for i in range(len(result) - 1):
        if pd.notna(result.at[i, code_col]) and pd.isna(result.at[i, amount_col]):
            next_row = i + 1
            while pd.isna(result.at[next_row, code_col]) and pd.notna(result.at[next_row, amount_col]):
                result.at[i, amount_col] = result.at[next_row, amount_col]
                rows_to_drop.append(next_row)
                next_row += 1

    # Drop rows where code was NaN and amount was moved to a previous row
    result = result.drop(rows_to_drop).reset_index(drop=True)

    return result


def preprocess_dataframe(result):
    # Use the applymap function to check each cell for English characters
    english_char_presence = result.applymap(has_english_chars)

    # Sum the boolean values along rows
    rowwise_english_count = english_char_presence.sum(axis=1)

    # Keep rows that have at least one English character
    result = result[rowwise_english_count > 0].reset_index(drop=True)

    # Drop non-English content
    result = result.applymap(filter_non_english)

    result = result.replace(to_replace='\n', value=' ', regex=True)

    col_name = result.columns[0]
    if not result[col_name].str.contains(r'(?:\d{4})').any():
        # Drop the column if the condition is False for all rows
        result.drop(columns=[col_name], inplace=True)

    col_name = result.columns[0]
    if any(result[col_name].str.contains(r'(?:\d{4})') & result[col_name].str.contains(r'\D') & result[
        col_name].str.contains(r'(?:[a-zA-Z])')):
        # Extract those first four digits and create a new column
        result.loc[:, 'code'] = result[col_name].str.extract(r'(\d{4})')[0]

        # Remove everything before and including the extracted digits from the original column
        result[col_name] = result[col_name].str.replace(r'.*?(\d{4})', '', 1, regex=True).str.strip()

        # Reorder columns to make 'code' the first column
        col_order = ['code'] + [col for col in result if col != 'code']
        result = result[col_order]
    result = handle_missing_values(result)

    # result = amount_in_char(result)

    # Identify the 'tax_amount' column
    tax_amount_col = identify_amount_column(result)

    # Convert the 'tax_amount' column to numeric (invalid parsing will be set as NaN)
    result[tax_amount_col] = pd.to_numeric(result[tax_amount_col], errors='coerce')

    # Drop rows where 'tax_amount' is NaN
    result = result.dropna(subset=[tax_amount_col])
    # if len(result.columns) == 5:
    #     result = result.drop(result.columns[0], axis=1)

    # Replace empty strings with NaN and then drop rows and columns with all NaN values
    result.replace('', pd.NA, inplace=True)

    # Remove rows and columns where all values are empty (or NaN after pivoting)
    result = result.dropna(axis=0, how='all').dropna(axis=1, how='all')

    # Reset row indices
    result = result.reset_index(drop=True)
    # Reset column names
    result.columns = ['code', 'tax_type', 'tax_amount', 'reason']

    # Drop rows where 'tax_amount' is NaN (empty)
    result = result.dropna(subset=['tax_amount'])

    # Extract the last 4 characters
    result['code'] = result['code'].astype(str).str[-4:]

    # Replace values that are not 4-digit numbers with NaN
    result.loc[~result['code'].str.match("^\d{4}$"), 'code'] = np.nan

    # Fill any remaining NaN values with empty strings
    result = result.fillna('')

    return result


def process_image_and_generate_dataframe(image_data, year):
    try:
        # Define the API endpoint for text extraction
        url = 'https://app.nanonets.com/api/v2/OCR/Model/dce4a973-1090-466e-b2d8-01e419ac182c/LabelFile/?async=false'

        # Send the image data to the OCR API
        data = {'file': ('image.png', image_data)}
        response = requests.post(url, auth=requests.auth.HTTPBasicAuth('12c3868c-37a9-11ee-901b-f20d4ed62372', ''),
                                 files=data)

        if response.status_code == 200:
            response_data = json.loads(response.text)
            df = pd.DataFrame(response_data['result'][0]['prediction'][0]['cells'])
            df = df.pivot(index='row', columns='col', values='text')
            df = preprocess_dataframe(df)
            df['YEAR'] = year
            return df

        else:
            # Handle the case when the OCR API response is not successful
            return pd.DataFrame()
    except Exception as e:
        # Handle exceptions, e.g., if the image cannot be processed
        return pd.DataFrame()

# Rest of your code remains unchanged
