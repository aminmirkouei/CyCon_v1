# CyCon tool

## Overview 
The application uses docker compose to orchestrate three containers or services:  a MYSQL database (db), a Flask application (backend) an NGINX proxy to route the request to the backend. The services communicate internally using docker networking and only the proxy is exposed to the internet. When a user sends a request, it is routed from the proxy to the backend.

The application has the following structure:
1. db: contains the database password and database schema files
2. backend: contains the Flask application code and a Dockerfile
3. proxy: contains the NGINX proxy configuration and a Dockerfile

## Development

The MySQL database requires a `password.txt` file to be present in the `db` folder. Create the `password.txt` file and put a password inside. Delete the MySQL docker volume `lca-tool_db-data` and restart the containers.

### Docker
For development, the easiest approach is to use the docker desktop application. After installation, any editor can be used to edit the code. However, to spawn a local development server to run the application locally, start the docker desktop application. Then, in a terminal prompt, run the command:

```bash
# inside the root folder of the repository, run the command:
docker compose up
```

### Vagrant 
For development with `vagrant`, use `vagrant up` in the root of the repository to build the VM.
Then use `vagrant ssh` to login to the VM. Finally, `cd /vagrant` and run `docker compose up` to start the containers:
```bash
vagrant up
vagrant ssh
cd vagrant
docker compose up
```
