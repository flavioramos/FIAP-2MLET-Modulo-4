import boto3
import os
from datetime import date, datetime
import yfinance as yf
import pandas as pd
import io
import pyarrow as pa
import pyarrow.parquet as pq

ssm_client = boto3.client('ssm')
s3 = boto3.client('s3')

# Variáveis de configuração (via AWS Systems Manager)
def get_variable(variable_name):
    return ssm_client.get_parameter(
        Name=variable_name, 
        WithDecryption=False
    )['Parameter']['Value']

# ARN do S3 onde o parquet será salvo
s3_parquet_target_uri = get_variable("s3_parquet_target_uri")

# Nome da ação
ticker = get_variable("ticker")

# Data inicial da primeira execução, fixa no AWS SM, no formato YYYY-MM-DD
first_run_start_date = datetime.strptime(get_variable("first_run_start_date"), '%Y-%m-%d').date()


def get_last_update():
    try:
        # Para testes, usamos uma data fixa. Caso deseje usar a variável do SSM, descomente a linha abaixo.
        # return datetime.strptime(get_variable("last_update"), '%Y-%m-%d').date()
        return datetime.strptime("2015-01-01", '%Y-%m-%d').date()
    except Exception:
        return ""


def set_last_update(end_date):
    ssm_client.put_parameter(
        Name="last_update",
        Value=end_date.strftime("%Y-%m-%d"),
        Type="String",
        Overwrite=True
    )


def download_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if df.empty:
        return {
            "statusCode": 200,
            "body": f"Nenhum dado retornado para {ticker} no período {start_date} - {end_date}."
        }
    
    # Reset index so that 'Date' becomes a regular column
    df.reset_index(inplace=True)
    
    # If the columns are MultiIndex, flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Ensure "Adj Close" exists
    if "Adj Close" not in df.columns:
        df["Adj Close"] = pd.NA
    
    # Convert the 'Date' column to string to avoid TimestampNTZ issues in AWS Glue
    df['Date'] = df['Date'].astype(str)
    
    # Debug: print the DataFrame columns
    print("Colunas do DataFrame:", list(df.columns))
    
    return df



def lambda_handler(event, context):
    start_date = get_last_update()
    end_date = date.today()

    if start_date == end_date:
        return {
            'statusCode': 200,
            'body': "Nothing to update"
        }

    print(f"s3 root: {s3_parquet_target_uri}")
    print(f"ticker: {ticker}")
    print(f"first run start date: {first_run_start_date}")
    print(f"start date: {start_date if start_date else 'EMPTY'}")

    if start_date == "":
        start_date = first_run_start_date
    
    print(f"total of days to be downloaded: {(end_date - start_date).days}")
    print("downloading...")

    result = download_data(ticker, start_date, end_date)

    if isinstance(result, dict) and result.get("statusCode") == 200:
        return {
            'statusCode': 500,
            'body': "Error downloading"
        }
    
    # Reordena as colunas para que coincidam exatamente com o esquema do PyArrow
    columns_order = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    result = result[columns_order]
    
    parquet_buffer = io.BytesIO()

    # Cria uma tabela PyArrow com o esquema esperado
    table = pa.Table.from_pandas(result, schema=pa.schema([
        pa.field('Date', pa.string()),
        pa.field('Open', pa.float64()),
        pa.field('High', pa.float64()),
        pa.field('Low', pa.float64()),
        pa.field('Close', pa.float64()),
        pa.field('Adj Close', pa.float64(), nullable=True),
        pa.field('Volume', pa.int64())
    ]), preserve_index=False)

    print("\n📦 Esquema do Parquet antes do upload:")
    print(table.schema)
    print("-" * 40)

    pq.write_table(table, parquet_buffer)
    parquet_buffer.seek(0)

    print("uploading...")
    s3.upload_fileobj(parquet_buffer, s3_parquet_target_uri, "raw/raw.parquet")

    set_last_update(end_date)

    return {
        'statusCode': 200,
        'body': f"S3 root: {s3_parquet_target_uri}, ticker: {ticker}, start_date: {start_date}, end_date: {end_date}"
    }


if os.getenv("LOCAL_ENV"):
    print(lambda_handler(None, None))
