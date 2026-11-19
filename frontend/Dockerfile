# ==============================================================================
# Production Dockerfile for CT Spleen Volumetry Web UI (Frontend Service)
# ==============================================================================
FROM nginx:1.27-alpine-slim

# Copy custom Nginx configuration with reverse proxy for API routes
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy web assets to Nginx default public directory
COPY index.html /usr/share/nginx/html/index.html

# Expose standard HTTP port
EXPOSE 80

# Healthcheck verifying Nginx is serving the single-page application
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://127.0.0.1:80/ || exit 1

# Start Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
