FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files and app code
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
