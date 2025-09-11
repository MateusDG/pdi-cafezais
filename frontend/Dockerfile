FROM node:20-alpine

WORKDIR /app

# ATENÇÃO: relativo ao contexto ./frontend
COPY package.json ./
# se tiver lock, copie também (não é obrigatório):
# COPY package-lock.json ./
RUN npm install

COPY . .

EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
