# Hiring_Platform


# To run django

1. Set `.env` with Azure Openai credentials

```
OPENAI_API_KEY = "<openai_key>"
HF_TOKEN = "<your_hf_token>"
```
2. Run uvicorn

```bash
cd reports
uvicorn response_mapping:app --port 7001
```

3. Run django : `python manage.py runserver`  
