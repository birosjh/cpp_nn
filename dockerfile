FROM gcc:9.5.0

WORKDIR /app

RUN apt update && apt -y install cmake protobuf-compiler