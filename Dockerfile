# Use the official Python image from the Docker Hub
#FROM sipltechnion/pytorch2.2.1_cuda12.1_ubuntu22.04
FROM nvcr.io/nvidia/pytorch:24.05-py3
ARG DEBIAN_FRONTEND=noninteractive
# Set environment variables to prevent Python from writing .pyc files and to buffer stdout and stderr
#ENV PYTHONDONTWRITEBYTECODE=1
#ENV PYTHONUNBUFFERED=1

# Create a directory for the app
#WORKDIR /app

# Copy the requirements file
#COPY requirements.txt /app/
COPY requirements.txt /root/requirements.txt

#RUN apt update && apt install -y git
RUN apt update -y

#RUN nvtop for gpu monitoring
RUN apt install nvtop 

# Install dependencies
RUN pip install --no-cache-dir -r /root/requirements.txt

RUN python -m spacy download en_core_web_sm

RUN apt install screen -y

RUN apt install ffmpeg -y

# Create a directory for the app
WORKDIR /app

# Copy the current directory contents into the container at /app
#COPY . /app/

# Expose port 8888 for JupyterLab
#EXPOSE 8888

# Run JupyterLab
#CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
