# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.4.0-base-ubuntu20.04

# Set timezone environment variable to avoid interactive prompt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Update and upgrade the system packages
RUN apt-get update && apt-get upgrade -y

# Install required dependencies for adding PPA and wget
RUN apt-get install -y software-properties-common wget git

# Install CMake (needed for dlib)
RUN apt-get install -y cmake

# Install build-essential (which includes C++ compiler and other necessary tools)
RUN apt-get install -y build-essential

# Add the deadsnakes PPA for newer Python versions
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update

# Install Python 3.12 and other necessary packages
RUN apt-get install -y python3.12 python3.12-venv python3.12-dev python3.12-distutils

# Set alternatives for Python 12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install build dependencies for dlib
RUN apt-get install -y cmake build-essential libopenblas-dev liblapack-dev

# Install pip for Python 12
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && rm get-pip.py

# Install the packaging and setuptools modules required by NVIDIA Apex
RUN pip install packaging setuptools

# Install torch before installing NVIDIA Apex
RUN pip install torch

# Install CUDA toolkit
RUN apt-get install -y cuda-toolkit-12-4

# Install NVIDIA Apex for mixed precision training (optional, if needed)
RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    python3 setup.py install --cuda_ext --cpp_ext

# Copy the requirements.txt file into the container and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
RUN pip install --upgrade opencv-python

# Copy the verify_gpu.py script into the container
COPY app/verify_gpu.py /app/

# Run the verify_gpu.py script to check GPU support
RUN python /app/verify_gpu.py

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the rest of the application code
COPY app /app

# Explicitly install face_recognition (if not included in requirements.txt)
RUN pip install face_recognition

# Install Flask
RUN pip install flask

# Expose any ports that your application uses (optional)
EXPOSE 5000

# Command to run your application
CMD ["python", "./trainaionimageanddescofpeople.py"]