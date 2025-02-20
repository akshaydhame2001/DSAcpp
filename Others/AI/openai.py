import openai
import os
from aws_lambda_powertools import Logger

logger = Logger()
openai.api_key = os.getenv("OPENAI_API_KEY")

def lambda_handler(event, context):
    message = event["body"]["message"]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4o"
        messages=[{"role": "user", "content": message}]
    )
    return {
        "statusCode": 200,
        "body": response.choices[0].message.content
    }