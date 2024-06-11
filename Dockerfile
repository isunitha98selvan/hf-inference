FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY app/hf_inference.py .

# Set the entry point to run the script
CMD ["python", "hf_inference.py"]