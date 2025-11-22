# Hiring_Platform


# To run response_mapping.py

1. Set `.env` with Azure Openai credentials

   ```
   AZURE_OPENAI_ENDPOINT = "https://<your_endpoint>.openai.azure.com/"
  AZURE_OPENAI_API_KEY = "<openai_key>"
  AZURE_OPENAI_DEPLOYMENT = "gpt-4.1"
  ```
2. Run `uvicorn response_mapping:app`
