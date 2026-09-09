FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV UPDOWN_DATA_DIR=/data \
    UPDOWN_REPORTS_DIR=/reports \
    PYTHONUNBUFFERED=1
CMD ["updown-collect"]
