# Imagem base com CUDA e cuDNN (Ubuntu 20.04)
FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

# Instala dependências do sistema e Python
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Define o diretório de trabalho dentro do container
WORKDIR /app

# Copia o arquivo de dependências e instala as bibliotecas necessárias
COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# Copia todo o código fonte para dentro do container
COPY . .

# Define as variáveis de ambiente para o Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expõe a porta 5000 (onde a aplicação Flask irá rodar)
EXPOSE 5000

# Comando para iniciar a aplicação Flask
CMD ["flask", "run"]
