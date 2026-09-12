FROM python:3.13.5-slim-bookworm
LABEL org.apex.runtime.protocol="1"
WORKDIR /opt/apex-runtime
COPY codec.py worker.py measurement.py ./
USER 65534:65534
ENTRYPOINT ["python", "-I", "-B", "/opt/apex-runtime/worker.py"]
