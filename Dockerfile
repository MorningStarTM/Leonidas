FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
