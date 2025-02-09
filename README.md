# Stock Prediction with LSTM e MLflow

Este projeto implementa uma aplicação para treinamento e previsão de preços de ações utilizando uma rede neural LSTM desenvolvida com PyTorch. A aplicação expõe uma API em Flask para disparar o treinamento, realizar predições e listar artefatos gerados, enquanto o MLflow é utilizado para rastreamento (tracking) dos experimentos de treinamento.

## Índice
- Características do Projeto
- Estrutura do Projeto
- Pré-requisitos
- Configuração e Execução
  - Usando Docker Compose
  - Execução Local
- Endpoints da API
- Parâmetros e Configurações
- Monitoramento com MLflow
- Contribuição
- Licença

## Características do Projeto


![Diagrama do Projeto](https://lh3.googleusercontent.com/d/1tnj2kYu3lI3pXmBH3kHeaiyqVh5Z7hoi)


**Treinamento de Modelo:**  
Utiliza dados históricos de ações (obtidos via Yahoo Finance) para treinar um modelo LSTM que realiza previsões de preços de fechamento.

**Predição:**  
A API permite que se realize predições para datas futuras (ou para comparar com dados históricos).

**Tracking com MLflow:**  
Registra os parâmetros, métricas (loss, MAE, RMSE, MAPE) e artefatos (modelo treinado, scaler, etc) para monitoramento dos experimentos.

**Containerização:**  
O projeto está preparado para execução via Docker, com dois serviços (Flask e MLflow) orquestrados por Docker Compose.

## Estrutura do Projeto

```
├── docker-compose.yaml            # Define os serviços (flask_model_server e mlflow_server)
├── .env                           # Variáveis de ambiente para o Docker Compose
├── flask_model_server/            
│   ├── app.py                     # Aplicação Flask com os endpoints (/train, /predict, /artifacts)
│   ├── config.py                  # Configura diretórios para artefatos, logs e parâmetros
│   ├── default_params.txt         # Parâmetros padrão (ticker, epochs, learning rate, etc)
│   ├── Dockerfile                 # Dockerfile para construir a imagem do servidor Flask
│   ├── local_run.sh               # Script para execução local da aplicação Flask
│   ├── local_setup.sh             # Script para setup do ambiente virtual local
│   ├── models/
│   │   └── lstm_model.py          # Implementação do modelo LSTM com PyTorch
│   ├── training/
│   │   ├── train.py               # Script para treinamento do modelo
│   │   └── predict.py             # Script para realizar predições
│   └── utils/
│       ├── config_loader.py       # Carrega os parâmetros de configuração
│       ├── data_utils.py          # Funções para download e pré-processamento de dados
│       └── file_utils.py          # Funções utilitárias para manipulação de arquivos
└── mlflow_server/
    ├── Dockerfile                 # Dockerfile para construir a imagem do servidor MLflow
    ├── local_run.sh               # Script para execução local do servidor MLflow
    ├── local_setup.sh             # Script para setup do ambiente virtual do MLflow
    └── requirements.txt           # Dependências do MLflow (ex: mlflow==2.17.1)
```

## Pré-requisitos

- Docker e Docker Compose (para execução via containers)
- Python 3.10+ (caso deseje rodar a aplicação localmente sem Docker)
- CUDA e cuDNN (opcional) – Para utilizar GPU durante o treinamento com PyTorch

## Configuração e Execução

### Usando Docker Compose

Instale o Docker e o Docker Compose caso ainda não os possua.

Na raiz do projeto, execute o comando:

```bash
docker-compose up --build
```

Esse comando irá:
- Construir e iniciar o container do `flask_model_server` (API Flask na porta 5000).
- Construir e iniciar o container do `mlflow_server` (servidor MLflow na porta 5001).

**Volumes:**  
Os volumes configurados garantem a persistência de:
- Artefatos de treinamento (modelo, scaler, arquivos de update)
- Logs do MLflow
- Parâmetros de configuração

### Execução Local

Caso prefira executar a aplicação sem Docker:

#### Para o Servidor Flask

Navegue até o diretório `flask_model_server`:

```bash
cd flask_model_server
```

Configure o ambiente virtual e instale as dependências:

```bash
./local_setup.sh
```

Inicie a aplicação localmente:

```bash
./local_run.sh
```

A aplicação estará disponível em:  
http://localhost:5000

#### Para o Servidor MLflow

Navegue até o diretório `mlflow_server`:

```bash
cd mlflow_server
```

Configure o ambiente virtual e instale as dependências:

```bash
./local_setup.sh
```

Inicie o servidor MLflow:

```bash
./local_run.sh
```

O MLflow estará disponível em:  
http://localhost:5001

## Endpoints da API

A aplicação Flask expõe os seguintes endpoints:

**GET /train**  
- **Descrição:** Inicia o processo de treinamento do modelo.  
- **Parâmetro Opcional:** `reset`  
  - Exemplo: `/train?reset=true` (para reiniciar o treinamento do zero definindo a data inicial).  
- **Retorno:** JSON contendo métricas do treinamento (loss, mae, rmse, mape).

**GET /predict**  
- **Descrição:** Realiza a predição do preço de fechamento para uma data específica.  
- **Parâmetro Obrigatório:** `date` no formato YYYY-MM-DD  
  - Exemplo: `/predict?date=2025-01-15`  
- **Retorno:** JSON com o valor predito e, se disponível, o valor real para comparação.

## Parâmetros e Configurações

**config.py:**  
Define os diretórios para:
- **ARTIFACTS_DIR:** Armazenamento dos artefatos (modelo, scaler, arquivos de update).
- **LOGS_DIR:** Logs do MLflow.
- **PARAMS_DIR:** Parâmetros do treinamento.

O script adapta os caminhos conforme o ambiente (local ou container).

**default_params.txt:**  
Contém os parâmetros padrão para o treinamento, como:
- `TICKER` – Símbolo da ação (ex: AAPL)
- `SEQUENCE_LENGTH` – Comprimento da sequência de entrada
- `EPOCHS` – Número de épocas de treinamento
- `LEARNING_RATE` – Taxa de aprendizado
- `HIDDEN_SIZE` – Número de neurônios na camada oculta
- `NUM_LAYERS` – Número de camadas LSTM
- `DATE_ZERO` – Data inicial para o treinamento

Para alterar os parâmetros, edite o arquivo `params.txt` que é copiado para o diretório de parâmetros conforme as configurações definidas.

## Monitoramento com MLflow

**Registro de Experimentos:**  
Durante o treinamento, os parâmetros, métricas e artefatos são registrados no MLflow.

**Acesso:**  
A interface do MLflow pode ser acessada através de http://localhost:5001.

Utilize esta interface para visualizar o histórico dos experimentos, comparações de métricas e download dos artefatos gerados.
