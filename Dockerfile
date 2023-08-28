# Use the official Python image as the build image
FROM python:latest

# Set the working directory

WORKDIR /root/mlapp

# Copy the requirements.txt file

COPY requirements.txt ./

# Install the dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files

COPY . .

EXPOSE 5000

#command that runs when container starts

CMD ["python","/root/mlapp/app3_Final.py"]

