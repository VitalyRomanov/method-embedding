FROM python:3.8-bullseye

WORKDIR /usr/src/app

RUN wget https://github.com/CoatiSoftware/Sourcetrail/releases/download/2020.1.117/Sourcetrail_2020_1_117_Linux_64bit.tar.gz
RUN tar -zxvf Sourcetrail_2020_1_117_Linux_64bit.tar.gz
ENV PATH="/usr/src/app/Sourcetrail":$PATH
RUN export LD_LIBRARY_PATH=/usr/src/app/Sourcetrail/lib:$LD_LIBRARY_PATH
ENV APP_PATH="/usr/src/app"

RUN python -m venv python_env
RUN pip install pandas==1.1.1
RUN apt-get update -y
RUN apt-get install -y sqlite3
RUN apt-get install -y libglx0

COPY sourcetrail_verify_files.py .
COPY process_folders.sh .

CMD ["bash", "process_folders.sh", "python_env", "/dataset"]