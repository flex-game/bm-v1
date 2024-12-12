FROM python:3.10-slim

WORKDIR /opt/ml/code

# Install only what we need
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy our code
COPY model_train.py .
COPY preprocessing.py .
COPY utils/ ./utils/

# Set up entrypoint
ENTRYPOINT ["python", "model_train.py"] 