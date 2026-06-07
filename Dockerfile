# Container for Hugging Face Spaces (Docker SDK) / any container host.
# HF Spaces routes to the port in README.md frontmatter (app_port: 7860).
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["python", "-m", "autosolver.web.server"]
