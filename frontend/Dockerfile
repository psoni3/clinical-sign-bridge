FROM node:22-alpine

WORKDIR /workspace
COPY package.json package-lock.json ./
RUN npm ci --legacy-peer-deps
COPY . .

EXPOSE 4200
CMD ["npm", "run", "start", "--", "--project", "patient", "--host", "0.0.0.0"]
