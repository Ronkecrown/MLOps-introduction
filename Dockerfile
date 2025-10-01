# Use a base image
FROM python:3.13.2

# set working directory
WORKDIR /app

# copy requirements and install
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser -m -s /bin/bash appuser

# Change ownership of the application directory to the non-root user
RUN chown -R appuser:appuser /app
# Switch to the non-root user
USER appuser

# Copy the rest of the app
COPY . /app

# Expose port for FastAPI
EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]