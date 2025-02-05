# Provedor AWS
provider "aws" {
  region = "us-east-1"  # Altere para sua região
}

# Role do IAM para o Lambda
resource "aws_iam_role" "lambda_exec_role" {
  name = "tickerDownloader-role-ka1crw0i-tf"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# Anexar política básica de execução para o Lambda (logs no CloudWatch)
resource "aws_iam_role_policy_attachment" "lambda_policy" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Função Lambda
resource "aws_lambda_function" "my_lambda" {
  function_name = "terraform-lambda-function"
  role          = aws_iam_role.lambda_exec_role.arn
  handler       = "lambda_function.lambda_handler"
  runtime       = "python3.9"
  filename      = "lambda_package.zip"  # O pacote ZIP do código
  source_code_hash = filebase64sha256("lambda_package.zip")  # Detecta alterações no código
  timeout       = 10  # Timeout em segundos
}

