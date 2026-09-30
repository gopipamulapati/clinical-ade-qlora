# Inference image (GPU host recommended). Mount or bake in the adapter directory.
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
ENV ADE_BASE=Qwen/Qwen2.5-1.5B-Instruct \
    ADE_ADAPTER=/models/adapter
EXPOSE 8000
CMD ["uvicorn", "ade_ft.serve:app", "--host", "0.0.0.0", "--port", "8000"]
