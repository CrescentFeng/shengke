FROM golang:1.24-alpine AS build

ARG MINIO_VERSION=RELEASE.2025-10-15T17-29-55Z
RUN apk add --no-cache git \
    && GOBIN=/out go install github.com/minio/minio@${MINIO_VERSION}

FROM alpine:3.22

RUN apk add --no-cache ca-certificates
COPY --from=build /out/minio /usr/local/bin/minio

EXPOSE 9000 9001
ENTRYPOINT ["minio"]
