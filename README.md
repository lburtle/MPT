# MPT (Market Prediction Tool) WIP

## Will come up with a better name later
## Realized there was no DAO involved in DAOv1.2

-----------------------------------

## Setup
### - Put in API_KEY=... with your NewsAPI key in docker-compose.yml for the environment (make sure to NOT use quotation marks)
### - In Dockerfile, put the correct path for the directory as WORKDIR
### - In the crontab file, put your correct path for python long with the path to the files in the repository
### - In CLI, type docker-compose up -d
### Everything should run hopefully
### --------------------------------------------
#### This current version runs two BNNs simultaneously
#### Consider your computation capacity before running
#### Can alternatively combine ticker lists into one BNN file to run only one at a time.
