#!/bin/bash

echo "📦 Empacotando código da função Lambda..."
zip -r lambda_package.zip lambda_function.py > /dev/null

echo "🚀 Iniciando o deploy com Terraform..."
terraform init
terraform apply -auto-approve

echo "✅ Deploy concluído com sucesso!"
