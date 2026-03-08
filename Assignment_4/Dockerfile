# Use lightweight Python base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy dependencies first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the rest of the app files
COPY . .

# Expose port from container to host
EXPOSE 5000

# Command to run the application using flask cli so it binds to 0.0.0.0
ENV FLASK_APP=app.py
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
