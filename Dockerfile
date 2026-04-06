# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Flask default)
EXPOSE 5000

# Run your application
CMD ["python", "application.py"]