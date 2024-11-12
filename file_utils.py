import os

def save_text_to_file(file_name, content):
    """Save text content to a file."""
    with open(file_name, 'w') as file:
        file.write(content)

def clean_up_files(*file_names):
    """Remove specified files."""
    for file_name in file_names:
        os.remove(file_name) 