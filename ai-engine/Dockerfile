FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY app ./app
COPY predict_video.py extract_include_landmarks.py ./
COPY src/clinicalsignbridge_ai/data/sequence_dataset.py \
    src/clinicalsignbridge_ai/data/sequence_normalizer.py \
    ./src/clinicalsignbridge_ai/data/
COPY src/clinicalsignbridge_ai/models/bilstm.py ./src/clinicalsignbridge_ai/models/

RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch==2.15.0+cpu" \
    && python -m pip install --no-cache-dir ".[inference]"

RUN addgroup --system app && adduser --system --ingroup app app
USER app

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
