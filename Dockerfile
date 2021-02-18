FROM python:3.8-buster

RUN apt-get update -y && apt-get install libgl1-mesa-glx -y

# Install python dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy contents
COPY . /app

WORKDIR /app

EXPOSE 8000

ENTRYPOINT ["uvicorn", "server:app", "--reload"]
CMD ["--host", "0.0.0.0"]

# Clean up
# docker system prune -a --volumes
