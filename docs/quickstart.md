# Quickstart

## Install
```
pip install -r requirements.txt
pip install .
```

## CLI
```
pacctl run-unified --mode auto
pacctl validate-entropy -n 10 --rows 100 --cols 5
```

## API
```
uvicorn services.api:app --reload
# GET /health, GET /metrics, POST /optimize
```
