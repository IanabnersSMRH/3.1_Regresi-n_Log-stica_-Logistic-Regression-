# 3.1_Logistic_Regression

# Práctica 3.1: Regresión Logística (`LogisticRegression`)

Práctica de análisis supervisado centrada en clasificación binaria mediante regresión logística y la función sigmoide (`predict` y `predict_proba`) en Python con `NumPy` y `scikit-learn`.

## Ejercicios incluidos (`LogiRegr.py`)

1. **Aprobación de crédito:** Predicción de aprobación (`0 = Rechazado`, `1 = Aprobado`) a partir de ingreso mensual y puntaje de historial crediticio.
2. **Detección de spam:** Clasificación de correos (`0 = Legítimo`, `1 = Spam`) según cantidad de enlaces y palabras clave de urgencia o premio.
3. **Riesgo de deserción estudiantil:** Estimación de riesgo de baja (`0 = Continúa`, `1 = Baja`) a partir de faltas acumuladas y promedio actual, revisando también los coeficientes e intercepto del modelo.

## Ejecución

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy scikit-learn
python3 LogiRegr.py
