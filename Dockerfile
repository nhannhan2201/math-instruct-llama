# Supply a GPU/PyTorch runtime image verified on the serving host.
# No CUDA image/version is chosen by this CPU implementation session.
ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE}
WORKDIR /app
COPY requirements.txt .
RUN python -c "import torch" && pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY app.py .
# Mount adapter/cache separately; set MATH_ADAPTER_PATH and MATH_METHOD.
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
