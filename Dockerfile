# AlphaToe — AlphaGumbel-only web deployment (Coolify-friendly).
#
# Builds a slim CPU-only image that serves the FastAPI UI in src/web/, with
# WEB_MODE=alphagumbel so the model selector is hidden and only the
# AlphaGumbel agent is registered.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime deps directly (the project's pyproject.toml pins CUDA
# wheels, which we don't want on a CPU host).
RUN pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        "torch>=2.0" \
        "numpy>=2.0" \
        "fastapi>=0.110" \
        "uvicorn[standard]>=0.27"

# Application code + the single checkpoint we need.
COPY src/ /app/src/
COPY models/cnn/cnn_c128b3_100k_reg_best.pt /app/models/cnn/cnn_c128b3_100k_reg_best.pt

ENV WEB_MODE=alphagumbel \
    PORT=8000

EXPOSE 8000

# Coolify supplies $PORT; fall back to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn web.server:app --host 0.0.0.0 --port ${PORT:-8000} --app-dir src"]
