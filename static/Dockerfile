FROM python:3.10.9

# 安装Anaconda
# SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh && \
    bash Anaconda3-2023.03-Linux-x86_64.sh -b -p /opt/anaconda && \
    rm Anaconda3-2023.03-Linux-x86_64.sh


# 添加环境文件
COPY environment.yaml .

# RUN conda config --set remote_connect_timeout_secs 600 && \
#     conda config --set remote_read_timeout_secs 600

# 创建新的环境
RUN /opt/anaconda/bin/conda env create -f environment.yaml -n deeplabcut --timeout 600

# 设置环境变量
ENV PATH /opt/anaconda/envs/deeplabcut/bin:$PATH

# 激活环境
SHELL ["conda", "run", "-n", "deeplabcut", "/bin/bash", "-c"]

WORKDIR /data
CMD [ "python", "app.py" ]
EXPOSE 80/tcp