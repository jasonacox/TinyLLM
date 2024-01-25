FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ARG VERSION=0.2.7
ENV PORT 8000

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt-get update && \
    apt-get -y --no-install-recommends install \
      python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip wheel
RUN python3 -m pip install --no-cache-dir vllm==${VERSION}

COPY entrypoint.sh /usr/local/bin/
CMD [ "entrypoint.sh" ]