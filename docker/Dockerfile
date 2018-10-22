FROM tensorflow/tensorflow:1.11.0-py3
MAINTAINER Yoshimura Naoya <yesyes708@gmail.com>

# Emacsのインストール
RUN apt-get update
RUN apt-get install emacs24-nox -y


# 作業ディレクトリ
RUN mkdir /root/work

# 追加python モジュール
COPY requirements.txt /root
RUN pip install -r /root/requirements.txt

WORKDIR /root
