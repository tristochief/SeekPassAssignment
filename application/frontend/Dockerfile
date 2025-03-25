# Stage 1: Build the React app
FROM node:14-alpine as build
WORKDIR /app
# Copy package files and install dependencies
COPY package*.json ./
RUN npm install
# Copy the rest of the code and build the app
COPY . .
RUN npm run build

# Stage 2: Serve the build with Nginx
FROM nginx:alpine
# Copy the production build from the previous stage
COPY --from=build /app/build /usr/share/nginx/html
# Copy our custom Nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
