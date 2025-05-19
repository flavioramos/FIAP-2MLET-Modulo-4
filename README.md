# Job Matching API usando Scikit‑Learn, Flask e MLflow

Este projeto implementa **um serviço de correspondência entre vagas e candidatos (job matching)**.
A solução utiliza **NLP com vetorização TF‑IDF** para transformar textos em features e **Regressão Logística** (com *Grid Search* para otimização do hiper‑parâmetro **C**) como classificador.
Todo o experimento é **rastreado pelo MLflow**, e a aplicação expõe uma **API REST em Flask** protegida por **autenticação JWT**.

> **Principais funcionalidades**
>
> * Treinamento do modelo a partir de três arquivos JSON (vagas, candidatos e prospects).
> * Predição da probabilidade de contratação para um candidato em determinada vaga.
> * Versionamento automático do modelo.
> * *Tracking* completo de métricas, parâmetros e artefatos no MLflow.
> * Execução simplificada via **Docker Compose** ou localmente em ambiente virtual.

---

## Índice

1. [Arquitetura de alto nível](#arquitetura-de-alto-nível)
2. [Estrutura do repositório](#estrutura-do-repositório)
3. [Pré‑requisitos](#pré-requisitos)
4. [Configuração e execução](#configuração-e-execução)
   1. [Usando Docker Compose](#usando-docker-compose)
   2. [Execução local](#execução-local)
5. [Endpoints da API](#endpoints-da-api)
6. [Parâmetros e personalização](#parâmetros-e-personalização)
7. [Monitoramento com MLflow](#monitoramento-com-mlflow)
8. [Testes automatizados](#testes-automatizados)
9. [Contribuição](#contribuição)
10. [Licença](#licença)

---

## Arquitetura de alto nível

![Diagrama do Projeto](https://lh3.googleusercontent.com/d/109wiog7azYJ41Gd0o0Os6uEbO2Xw6kET)

* **API Flask ** – expõe rota de login para obter token JWT e rotas protegidas para *train* e *predict*.
* **Camada de ML** – pipeline de pré‑processamento + classificador definido em `models/job_matching_model.py`.
* **Persistência** – modelo treinado, scaler e arquivos auxiliares são armazenados em volume compartilhado; cada versão recebe sufixo `_vN`.
* **Observabilidade** – métricas e artefatos são enviados ao MLflow, acessível via navegador.

---

## Estrutura do repositório

```
├── docker-compose.yaml            # Orquestra API Flask e MLflow
├── .env                           # Variáveis para Docker Compose
├── flask_model_server/
│   ├── app.py                     # API Flask (JWT, /train, /predict)
│   ├── config.py                  # Diretórios, mapeamentos, hiper‑parâmetros
│   ├── default_params.txt         # Credenciais padrão + segredos JWT
│   ├── Dockerfile                 # Imagem da API
│   ├── local_run.sh               # Execução local
│   ├── local_setup.sh             # Criação do venv local
│   ├── models/
│   │   └── job_matching_model.py  # Pipeline TF‑IDF + LogisticRegression
│   ├── training/
│   │   ├── train.py               # Treinamento + GridSearch + MLflow
│   │   └── predict.py             # Função de predição
│   ├── utils/
│   │   ├── config_loader.py       # Carrega / persiste parâmetros
│   │   └── model_versioning.py    # Versionamento de modelos
│   └── tests/                     # Pytest para API e lógica
└── mlflow_server/                 # Container e scripts do MLflow
    ├── Dockerfile
    ├── local_run.sh
    ├── local_setup.sh
    └── requirements.txt
```

---

## Pré‑requisitos

* **Docker + Docker Compose** (recomendado)
  ou **Python 3.10+** para execução local.

---

## Configuração e execução

### Usando Docker Compose

```bash
# Na raiz do projeto
docker compose up --build
```

* `flask_model_server` → [http://localhost:5000](http://localhost:5000)
* `mlflow_server`     → [http://localhost:5001](http://localhost:5001)

Volumes persistem:

* Artefatos do modelo (`training_artifacts/`)
* Logs do MLflow (`mlflow_logs/`)
* Parâmetros da aplicação (`parameters/params.txt`)

### Execução local

```bash
# API Flask
cd flask_model_server
./local_setup.sh   # cria venv e instala deps
./local_run.sh     # inicia em http://localhost:5000

# MLflow (opcional)
cd ../mlflow_server
./local_setup.sh
./local_run.sh     # inicia em http://localhost:5001
```

> Defina a variável `LOCAL_RUN=true` para que `config.py` utilize caminhos locais.

---

## Endpoints da API

| Método & Rota     | Descrição                             | Parâmetros                                                                 | Exemplo                                                            |
| ----------------- | ------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **POST /login**   | Gera token JWT                        | `username`, `password`                                                     | `/login` com JSON `{ "username": "user", "password": "password" }` |
| **GET /train**    | Inicia o treinamento do modelo        | *Nenhum*                                                                   | `GET /train` (header `Authorization: Bearer <token>`)              |
| **POST /predict** | Prediz a probabilidade de contratação | `principais_atividades`, `competencia_tecnicas_e_comportamentais`, `cv_pt` | `/predict` com JSON                                                |

### Exemplo de *payload* para `/predict`

```json
{
  "principais_atividades": "Desenvolver APIs REST em Python e Flask",
  "competencia_tecnicas_e_comportamentais": "Experiência com Docker, AWS, boas práticas de código",
  "cv_pt": "Engenheiro de software com 6 anos de experiência em Python, Flask, AWS e ML"
}
```

**Resposta**

```json
{
  "probability": 0.87,
  "prediction": "Contratado",
  "confidence": 0.74,
  "model_version": 3
}
```

---

## Parâmetros e personalização

Todos os parâmetros estão em `flask_model_server/config.py` ou em `default_params.txt` (copiado para `parameters/params.txt` na primeira execução).
Altere valores como:

* `TFIDF_JOB_DESCRIPTION_MAX_FEATURES`, `TFIDF_CANDIDATE_CV_MAX_FEATURES`
* `GRID_SEARCH_C_VALUES`, `GRID_SEARCH_CV`
* Credenciais padrão e `JWT_SECRET_KEY`

Para persistir alterações, edite `parameters/params.txt` (carregado por `config_loader.py`).

---

## Monitoramento com MLflow

* **Experimentos** – Cada execução do `/train` cria um *run* no MLflow com métricas (AUC, precisão, recall, F1, etc.), hiper‑parâmetros e artefatos.
* **Download de modelos** – A interface web permite baixar qualquer versão treinada.

Acesse em \`http://localhost:5001\`.

---

## Testes automatizados

Rodar **pytest** na pasta `flask_model_server/tests`:

```bash
pytest -q
```

Os testes cobrem:

* Autenticação (`/login`)
* Proteção de rotas (`/train`, `/predict`)
* Funcionalidade de treino e predição (mocks)

---

## Licença

Distribuído sob a licença MIT. Consulte `LICENSE` para mais detalhes.
