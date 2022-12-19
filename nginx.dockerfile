FROM nginx:1.22.0
# Remove default Nginx config
RUN rm /etc/nginx/conf.d/default.conf
# RUN rm /etc/nginx/nginx.conf
# Copy proxy config
COPY ./nginx.conf /etc/nginx/conf.d/