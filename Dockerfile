FROM golang:1.25-alpine AS builder

WORKDIR /src

# Cache dependencies
COPY go.mod go.sum ./
RUN go mod download

# Build
COPY *.go ./
RUN CGO_ENABLED=0 go build -o /ask-pdf .

# -----------------------------------------------------------
FROM alpine:3.21

RUN apk add --no-cache ca-certificates tzdata

WORKDIR /app
COPY --from=builder /ask-pdf .
COPY prompts/ ./prompts/
COPY static/  ./static/

RUN mkdir -p uploads data

EXPOSE 3400

ENTRYPOINT ["./ask-pdf"]
