import os
import pandas as pd

def load_data(file_path):
    """Load data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    return pd.read_csv(file_path)

def clean_data(df):
    """Perform basic data cleaning."""
    # Example: Drop missing values
    df = df.dropna()
    # Add more cleaning steps as needed
    return df

def prepare_data(file_path):
    """Load and clean data."""
    df = load_data(file_path)
    df = clean_data(df)
    return df

if __name__ == "__main__":
    # Example usage
    data_file = 'data/sample_data.csv'  # Update with your actual data file path
    data = prepare_data(data_file)
    print(data.head()) 