FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    ARTIFACT_DIR=/app/artifacts \
    REPORT_DIR=/app/reports

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY playstyle ./playstyle
COPY webapp ./webapp
COPY data ./data
COPY wsgi.py .

# Train at build time so the image starts serving immediately.
RUN python -m playstyle.cli train --k 10

EXPOSE 8000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "120", "wsgi:app"]
