import json
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

THRESHOLD = 0.5
SUPERVISED_TYPES = {"TRANSFER", "CASH_OUT"}
PaySimType = Literal["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]

app = FastAPI(
    title="Labris Fraud Detection API (PaySim Based)",
    description="API de Detecção de Fraudes financeiras, baseada em um modelo treinado no clássico PaySim",
    version="0.1.0",
)

bundle = joblib.load("artifacts/xgb_fraud_model.joblib")
model = bundle["model"] if isinstance(bundle, dict) else bundle

with open("artifacts/feature_cols.json", encoding="utf-8") as f:
    schema_data = json.load(f)
feature_cols = schema_data["features"] if isinstance(schema_data, dict) else schema_data


class TransactionPayload(BaseModel):
    step: int = Field(..., ge=1, description="Unidade de tempo em horas da simulação PaySim")
    type: PaySimType = Field(..., description="Tipo de transação no PaySim")
    amount: float = Field(..., gt=0, description="Valor monetário da transação")
    oldbalanceOrg: float = Field(..., ge=0, description="Saldo da conta de origem antes da transação")
    newbalanceOrig: float = Field(..., ge=0, description="Saldo da conta de origem após a transação")
    oldbalanceDest: float = Field(..., ge=0, description="Saldo da conta de destino antes da transação")
    newbalanceDest: float = Field(..., ge=0, description="Saldo da conta de destino após a transação")

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().upper()
        return value


def build_features_df(payload: TransactionPayload) -> pd.DataFrame:
    hour = payload.step % 24
    day_of_month = payload.step // 24
    error_orig = payload.newbalanceOrig + payload.amount - payload.oldbalanceOrg
    error_dest = payload.oldbalanceDest + payload.amount - payload.newbalanceDest
    dest_zero_flag = int((payload.oldbalanceDest == 0) and (payload.newbalanceDest == 0))
    type_transfer = int(payload.type == "TRANSFER")

    row = {
        "step": payload.step,
        "amount": payload.amount,
        "oldbalanceOrg": payload.oldbalanceOrg,
        "newbalanceOrig": payload.newbalanceOrig,
        "oldbalanceDest": payload.oldbalanceDest,
        "newbalanceDest": payload.newbalanceDest,
        "hour": hour,
        "day_of_month": day_of_month,
        "error_orig": error_orig,
        "error_dest": error_dest,
        "dest_zero_flag": dest_zero_flag,
        "type_TRANSFER": type_transfer,
    }
    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


@app.get("/")
def instrucoes() -> dict:
    return {
        "mensagem": "API de detecção de fraudes Labris (PaySim)",
        "tipos_aceitos_entrada": ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
        "tipos_suportados_modelo": ["TRANSFER", "CASH_OUT"],
        "threshold": THRESHOLD,
        "endpoints": {
            "instrucoes": "GET /",
            "status": "GET /status",
            "predicao": "POST /predict",
        },
        "exemplo_payload": {
            "step": 1,
            "type": "TRANSFER",
            "amount": 181.0,
            "oldbalanceOrg": 181.0,
            "newbalanceOrig": 0.0,
            "oldbalanceDest": 0.0,
            "newbalanceDest": 0.0,
        },
    }


@app.get("/status")
def status() -> dict:
    return {
        "status": "ok",
        "servico": "paysim-fraud-detection-api",
        "model_loaded": model is not None,
    }


@app.post("/predict")
def revisar_transacao(payload: TransactionPayload) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")

    if payload.type not in SUPERVISED_TYPES:
        return {
            "type": payload.type,
            "amount": payload.amount,
            "status": "out_of_scope",
            "fraud_probability": None,
            "is_fraud": None,
            "threshold": THRESHOLD,
        }

    X = build_features_df(payload)
    prob = float(model.predict_proba(X)[0][1])

    return {
        "type": payload.type,
        "amount": payload.amount,
        "status": "scored",
        "fraud_probability": round(prob, 6),
        "is_fraud": prob >= THRESHOLD,
        "threshold": THRESHOLD,
    }