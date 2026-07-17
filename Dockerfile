FROM python:3.12-slim

WORKDIR /app

COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

COPY src/ src/
COPY static/ static/
# The public landing page uses the project logo and visual assets.
COPY visual/ static/visual/
COPY models/checkpoints/best_model.pt models/checkpoints/best_model.pt

EXPOSE 8000

CMD ["uvicorn", "src.cloud.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
