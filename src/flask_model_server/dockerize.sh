#!/bin/bash
# dockerize.sh: Script para dockerizar a aplicação Flask com suporte a GPU,
# utilizando volumes Docker dedicados para armazenar os logs do MLflow e os artefatos do treinamento.

# Nome da imagem Docker que será criada
IMAGE_NAME="flask"

# Nome dos volumes Docker
VOLUME_LOGS="mlflow_logs"
VOLUME_ARTIFACTS="training_artifacts"

##############################
# Criação dos volumes Docker
##############################
# if ! docker volume ls --format '{{.Name}}' | grep -q "^${VOLUME_LOGS}$"; then
#     echo "Criando volume Docker '$VOLUME_LOGS' para os logs do MLflow..."
#     docker volume create $VOLUME_LOGS
# else
#     echo "Volume '$VOLUME_LOGS' já existe."
# fi

# if ! docker volume ls --format '{{.Name}}' | grep -q "^${VOLUME_ARTIFACTS}$"; then
#     echo "Criando volume Docker '$VOLUME_ARTIFACTS' para os artefatos do treinamento..."
#     docker volume create $VOLUME_ARTIFACTS
# else
#     echo "Volume '$VOLUME_ARTIFACTS' já existe."
# fi

##############################
# Criação do Dockerfile (se não existir)
##############################
# if [ ! -f Dockerfile ]; then
#     echo "Criando o Dockerfile..."
#     cat << 'EOF' > Dockerfile
# # Imagem base com CUDA e cuDNN (Ubuntu 20.04)
# FROM nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04

# # Instala dependências do sistema e Python
# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-pip \
#     && rm -rf /var/lib/apt/lists/*

# # Define o diretório de trabalho dentro do container
# WORKDIR /app

# # Copia o arquivo de dependências e instala as bibliotecas necessárias
# COPY requirements.txt .
# RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

# # Copia todo o código fonte para dentro do container
# COPY . .

# # Define as variáveis de ambiente para o Flask
# ENV FLASK_APP=app.py
# ENV FLASK_RUN_HOST=0.0.0.0

# # Expõe a porta 5000 (onde a aplicação Flask irá rodar)
# EXPOSE 5000

# # Comando para iniciar a aplicação Flask
# CMD ["flask", "run"]
# EOF
#     echo "Dockerfile criado com sucesso."
# else
#     echo "Dockerfile já existe. Utilizando o arquivo existente."
# fi

##############################
# Construção da imagem Docker
##############################
# echo "Construindo a imagem Docker '$IMAGE_NAME'..."
# docker build -t $IMAGE_NAME .

# if [ $? -ne 0 ]; then
#     echo "Erro na construção da imagem Docker."
#     exit 1
# fi

##############################
# Execução do container com acesso à GPU e volumes montados
##############################
echo "Executando o container com acesso à GPU..."
docker run --gpus all -d -p 5000:5000 \
  -v $VOLUME_ARTIFACTS:/app/training_artifacts \
  -v $VOLUME_LOGS:/app/mlflow_logs \
  $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo "Container iniciado com sucesso."
    echo "A aplicação Flask está disponível em http://localhost:5000"
else
    echo "Erro ao iniciar o container."
fi
