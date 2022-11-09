FROM node:16-alpine 

WORKDIR /user/src/app 

ENV PATH /user/src/app/node_modules/.bin:$PATH

COPY ./package.json ./

COPY ./src ./src

COPY ./public ./public

RUN npm install -g npm@latest 

RUN rm -rf node_modules

RUN npm install 

COPY . .

EXPOSE 3000

CMD ["npm","start"]

