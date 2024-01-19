# Backend Folder

Files pertaining to the all the instramentation and code performed in the backend or unseen by the user.

## Folders

Static:  <br />
templates: Contains the files related to the webpage html GUI templates that the user sees and interacts with.

## Individual files.

Dockerfile: The dockerfile to quickly assemble the backend files. <br>
__init__.py: Simple initiation file used during initial setup. <br>
**app.py**: First run Python file of the website. Imports all necessary files including the html code and constructs the overall website. <br>
**auth.py**: Python file containing all callable methods related to authenticating a user. Example: registering a user in the user table. <br>
config.py: Python file for constant variables and secret values for communication with the network. <br>
db.py: Python file that creates a connection to the database and methods for interacting with said database. Example: add new user to user table. <br>
experiments.py: Python code pertaining to the possible chooses for experiments. <br>
home.py: Python code related to the directory for the home page. This is set to index.html currently. <br>
mail.py: Python code related to the automatic emails sent by the website manager. <br>
requirements.txt: imported files and versions the website uses. <br>
utils.py: Utility information and operation. Example get current user information and set to active. <br>
