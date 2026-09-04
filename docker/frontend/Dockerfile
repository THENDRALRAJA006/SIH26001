# ── Build stage ──────────────────────────────────────────────────────
FROM node:20-alpine AS builder

LABEL maintainer="ZAIX Team"
LABEL project="LAND-JEPA"

WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ .
RUN npm run build

# ── Production stage ─────────────────────────────────────────────────
FROM nginx:1.25-alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY docker/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
