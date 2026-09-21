FROM python:3.12-slim
WORKDIR /app
COPY bh_classifier /app/bh_classifier
ENV PYTHONUNBUFFERED=1 PORT=8080
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2)"
CMD ["python", "-m", "bh_classifier.service"]
