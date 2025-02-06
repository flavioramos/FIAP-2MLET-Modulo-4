output "lambda_function_name" {
  description = "Nome da função Lambda"
  value       = aws_lambda_function.my_lambda.function_name
}

output "lambda_arn" {
  description = "ARN da função Lambda"
  value       = aws_lambda_function.my_lambda.arn
}
