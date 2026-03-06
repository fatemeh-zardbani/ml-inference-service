# base image
FROM python:3.12-slim

# set workdir
WORKDIR /app

# copy requirements early for caching
COPY requirements.txt ./

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . /app

# set environment variables
ENV MODEL_SOURCE=local
ENV MODEL_PATH=/app/app/model_artifacts

# ensure model artifacts exist (optional; smoke_test will create on first run)
# ENV PYTHONPATH=/app

# expose port
EXPOSE 8000

# default command to run uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
