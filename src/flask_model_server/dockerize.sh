#!/bin/bash
# dockerize.sh: Script para dockerizar a aplicação Flask com suporte a GPU,
# utilizando volumes Docker dedicados para armazenar os logs do MLflow e os artefatos do treinamento.

# Nome da imagem Docker que será criada
IMAGE_NAME="flask"

# Nome dos volumes Docker
VOLUME_LOGS="mlflow_logs"
VOLUME_ARTIFACTS="training_artifacts"

echo "Executando o container com acesso à GPU..."
docker run --gpus all -d -p 5000:5000 \
  -v mlflow_logs:/artifacts/training_artifacts \
  -v training_artifacts:/artifacts/mlflow_logs \
  $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo "Container iniciado com sucesso."
    echo "A aplicação Flask está disponível em http://localhost:5000"
else
    echo "Erro ao iniciar o container."
fi
