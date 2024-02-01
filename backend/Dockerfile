# syntax=docker/dockerfile:1.4
FROM --platform=$BUILDPLATFORM python:3.10-slim AS builder

WORKDIR /backend

COPY requirements.txt /backend

RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y libgomp1

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/playwright \
    playwright install

COPY . .

ENV FLASK_APP app.py
ENV FLASK_DEBUG 1
ENV FLASK_RUN_PORT 8000
ENV FLASK_RUN_HOST 0.0.0.0

EXPOSE 8000

CMD ["flask", "run"]

FROM builder AS dev-envs

RUN <<EOF
apk update
apk add git
EOF

RUN mkdir -p data

RUN <<EOF
addgroup -S docker
adduser -S --shell /bin/bash --ingroup docker vscode
EOF

# install Docker tools (cli, buildx, compose)
COPY --from=gloursdocker/docker / /

CMD ["flask", "run"]
