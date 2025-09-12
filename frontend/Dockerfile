# Multi-stage build for production optimization
FROM node:20-alpine AS development

WORKDIR /app

# Copy package files for better caching
COPY package.json package-lock.json* ./

# Install dependencies (fallback to npm install if no lock file)
RUN npm ci 2>/dev/null || npm install

# Copy source code
COPY . .

# Development command
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]

# Production build stage
FROM node:20-alpine AS build

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies including dev dependencies for build
RUN npm ci 2>/dev/null || npm install

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage with nginx
FROM nginx:alpine AS production

# Install curl for health checks
RUN apk add --no-cache curl

# Copy built assets to nginx
COPY --from=build /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Configure nginx to run as nginx user (built-in)
# No need for custom user, use nginx defaults

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:80/ || exit 1

EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
