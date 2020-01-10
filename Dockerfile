FROM python:3.7

COPY ./python-env/ /python-env/
RUN pip install -r /python-env/requirements.txt

EXPOSE 8554

#CMD ["python3", "/python-env/main.py"]