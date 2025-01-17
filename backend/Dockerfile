# Use a lightweight Python image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the backend files into the container
COPY . /app

# Copy the requirements.txt file into the container
COPY ./requirements.txt /app/requirements.txt

# Set up a virtual environment inside the container
RUN python -m venv .venv
RUN ./.venv/bin/pip install --no-cache-dir -r /app/requirements.txt

# Expose the port for the app (adjust if needed)
EXPOSE 5000

# Run the app (replace with your app entry point if needed)
CMD ["/app/.venv/bin/python", "main.py"]
