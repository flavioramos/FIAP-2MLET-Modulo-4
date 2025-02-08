import boto3
import os
import io
from datetime import date, datetime
import yfinance as yf
import pandas as pd

# Inicialização dos clientes AWS
ssm_client = boto3.client("ssm")
s3 = boto3.client("s3")

# Variáveis de configuração (via AWS Systems Manager)
def get_variable(variable_name):
    """ Obtém o valor de uma variável armazenada no AWS Systems Manager (SSM). """
    return ssm_client.get_parameter(
        Name=variable_name,
        WithDecryption=False
    )["Parameter"]["Value"]


# ARN do S3 onde o parquet será salvo
s3_parquet_target_uri = get_variable("s3_parquet_target_uri")

# Nome da ação
ticker = get_variable("ticker")

# Data inicial da primeira execução, fixa no AWS SSM, no formato YYYY-MM-DD
first_run_start_date = datetime.strptime(
    get_variable("first_run_start_date"), "%Y-%m-%d"
).date()


def get_last_update():
    """
    Obtém a última data de atualização do SSM.
    Para testes, usamos uma data fixa. 
    Caso deseje usar a variável do SSM, descomente a linha correspondente.
    """
    try:
        # return datetime.strptime(get_variable("last_update"), "%Y-%m-%d").date()
        return datetime.strptime("2015-01-01", "%Y-%m-%d").date()
    except Exception:
        return ""


def set_last_update(end_date):
    """ Atualiza a última data de execução no AWS SSM. """
    ssm_client.put_parameter(
        Name="last_update",
        Value=end_date.strftime("%Y-%m-%d"),
        Type="String",
        Overwrite=True
    )


def download_data(ticker, start_date, end_date):
    """
    Faz o download dos dados históricos de uma ação via Yahoo Finance.
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if df.empty:
        return {
            "statusCode": 200,
            "body": f"Nenhum dado retornado para {ticker} no período {start_date} - {end_date}."
        }

    return df


def lambda_handler(event, context):
    """
    Função principal do Lambda para baixar e armazenar os dados históricos 
    de uma ação no S3 em formato Parquet.
    """
    start_date = get_last_update()
    end_date = date.today()

    if start_date == end_date:
        return {
            "statusCode": 200,
            "body": "Nothing to update"
        }

    # Logs informativos
    print(f"s3 root: {s3_parquet_target_uri}")
    print(f"ticker: {ticker}")
    print(f"first run start date: {first_run_start_date}")
    print(f"start date: {start_date if start_date else 'EMPTY'}")

    if start_date == "":
        start_date = first_run_start_date

    print(f"Total de dias a serem baixados: {(end_date - start_date).days}")
    print("Downloading...")

    result = download_data(ticker, start_date, end_date)


    # 📌 Se o DataFrame retornado tem MultiIndex, resetamos
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [col[0] for col in result.columns]  # Mantém apenas o nome das colunas

    # 📌 Se 'Date' não estiver no DataFrame, resetamos o índice
    if "Date" not in result.columns:
        if isinstance(result.index, pd.DatetimeIndex):
            print("⚠️ 'Date' não encontrado como coluna! Usando índice como data.")
            result = result.reset_index()  # Transforma o índice em coluna
        else:
            raise KeyError("🚨 ERRO: A coluna 'Date' não está presente no DataFrame!")

    # Converter a coluna 'Date' para datetime e defini-la como índice
    result["Date"] = pd.to_datetime(result["Date"])
    result.set_index("Date", inplace=True)

    print("\n✅ Estrutura final do DataFrame:")
    print(result.head())


    if isinstance(result, dict) and result.get("statusCode") == 200:
        return {
            "statusCode": 500,
            "body": "Error downloading"
        }

    # Convertendo para Parquet e enviando para o S3
    parquet_buffer = io.BytesIO()
    result.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)

    print("Uploading...")
    s3.upload_fileobj(parquet_buffer, s3_parquet_target_uri, "raw/raw.parquet")

    # Atualizando a última execução
    set_last_update(end_date)

    return {
        "statusCode": 200,
        "body": f"S3 root: {s3_parquet_target_uri}, ticker: {ticker}, "
                f"start_date: {start_date}, end_date: {end_date}"
    }


# Execução local para testes
if os.getenv("LOCAL_ENV"):
    print(lambda_handler(None, None))
