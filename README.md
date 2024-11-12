# Buddy Model v1, BM-v1

## Setup Instructions

### Prerequisites

- Install a terminal app like a-Shell or iSH on your iOS device.
- Ensure Python 3 is installed.

### Virtual Environment Setup

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv myenv
   ```

2. **Activate the Virtual Environment**:
   ```bash
   source myenv/bin/activate
   ```

3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

### Environment Variables

- Create a `.env` file in the root directory.
- Add your environment variables in the following format:
  ```plaintext
  SECRET_KEY=your_secret_key
  DEBUG=True
  ```

### Deactivate the Virtual Environment

- To deactivate the virtual environment, simply run:
  ```bash
  deactivate
  ```

## Usage

- Activate the virtual environment before running your scripts.
- Use `python your_script.py` to execute your Python scripts.
