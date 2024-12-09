# Use a Python base image
FROM python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y curl

# Set the working directory
WORKDIR /app

# Copy the local code to the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY clearml.conf /root/clearml.conf

RUN clearml-agent init

RUN clearml-agent daemon --queue default &

# Expose the port for Gradio
EXPOSE 7860

ENTRYPOINT python RAG.py
