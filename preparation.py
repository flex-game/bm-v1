import os
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def authenticate_gdrive():
    """Authenticate and return a Google Sheets client."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
    client = gspread.authorize(creds)
    return client

def load_data_from_gsheet(sheet_url):
    """Load data from a Google Sheet."""
    client = authenticate_gdrive()
    sheet = client.open_by_url(sheet_url)
    worksheet = sheet.get_worksheet(0)  # Assuming data is in the first sheet
    data = worksheet.get_all_records()
    return pd.DataFrame(data)

def load_data(file_path=None, sheet_url=None):
    """Load data from a CSV file or Google Sheet."""
    if file_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        return pd.read_csv(file_path)
    elif sheet_url:
        return load_data_from_gsheet(sheet_url)
    else:
        raise ValueError("Either file_path or sheet_url must be provided.")

def clean_data(df):
    """Perform basic data cleaning."""
    # Example: Drop missing values
    df = df.dropna()
    # Add more cleaning steps as needed
    return df

def prepare_data(file_path=None, sheet_url=None):
    """Load and clean data."""
    df = load_data(file_path, sheet_url)
    df = clean_data(df)
    return df

if __name__ == "__main__":
    # Example usage
    data_file = 'data/sample_data.csv'  # Update with your actual data file path
    sheet_url = 'https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID/edit'  # Update with your actual sheet URL

    # Load data from CSV
    data = prepare_data(file_path=data_file)
    print(data.head())

    # Load data from Google Sheets
    data_from_gsheet = prepare_data(sheet_url=sheet_url)
    print(data_from_gsheet.head()) 