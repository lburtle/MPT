FROM python:3.9.6
WORKDIR /Projects/MPT
RUN apt-get update && apt-get install -y cron

COPY BNN_train.py .
COPY requirements.txt .
COPY Sentiment/sentimenttool.py .
COPY Sentiment/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY crontab /etc/cron.d/my-cron-jobs

RUN touch /var/log/cron.log
RUN chmod 0666 /var/log/cron.log

CMD cron && tail -f /var/log/cron.log