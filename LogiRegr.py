import numpy as np
from sklearn.linear_model import LogisticRegression

# ==========================================
# EJERCICIO 1: APROBACIÓN DE CRÉDITO
# ==========================================
X_credito = np.array([
    [12, 2],
    [15, 3],
    [18, 4],
    [35, 7],
    [42, 8],
    [50, 9],
])
y_credito = np.array([0, 0, 0, 1, 1, 1])

modelo_credito = LogisticRegression()
modelo_credito.fit(X_credito, y_credito)

nuevo_solicitante = np.array([[26, 6]])
pred_clase_cred = modelo_credito.predict(nuevo_solicitante)
pred_prob_cred = modelo_credito.predict_proba(nuevo_solicitante)

print("=== 1. APROBACIÓN DE CRÉDITO ===")
print(f"Predicción final: Clase {pred_clase_cred[0]} (0=Rechazado, 1=Aprobado)")
print(
    f"Probabilidades: P(Rechazado)={pred_prob_cred[0][0]*100:.1f}%, "
    f"P(Aprobado)={pred_prob_cred[0][1]*100:.1f}%\n"
)

# ==========================================
# EJERCICIO 2: DETECCIÓN DE SPAM
# ==========================================
X_spam = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [5, 4],
    [7, 6],
    [10, 8],
])
y_spam = np.array([0, 0, 0, 1, 1, 1])

modelo_spam = LogisticRegression()
modelo_spam.fit(X_spam, y_spam)

nuevo_correo = np.array([[3, 2]])
pred_clase_spam = modelo_spam.predict(nuevo_correo)
pred_prob_spam = modelo_spam.predict_proba(nuevo_correo)

print("=== 2. DETECCIÓN DE SPAM ===")
print(f"Predicción final: Clase {pred_clase_spam[0]} (0=Legítimo, 1=Spam)")
print(
    f"Probabilidades: P(Legítimo)={pred_prob_spam[0][0]*100:.1f}%, "
    f"P(Spam)={pred_prob_spam[0][1]*100:.1f}%\n"
)

# ==========================================
# EJERCICIO 3: RIESGO DE DESERCIÓN ESTUDIANTIL
# ==========================================
X_riesgo = np.array([
    [1, 9.2],
    [2, 8.8],
    [3, 8.4],
    [10, 6.2],
    [13, 5.8],
    [16, 5.1],
])
y_riesgo = np.array([0, 0, 0, 1, 1, 1])

modelo_riesgo = LogisticRegression()
modelo_riesgo.fit(X_riesgo, y_riesgo)

alumno_nuevo = np.array([[6, 7.5]])
pred_clase_riesgo = modelo_riesgo.predict(alumno_nuevo)
pred_prob_riesgo = modelo_riesgo.predict_proba(alumno_nuevo)

print("=== 3. RIESGO DE DESERCIÓN ===")
print(f"Predicción final: Clase {pred_clase_riesgo[0]} (0=Continúa, 1=Baja)")
print(
    f"Probabilidades: P(Continúa)={pred_prob_riesgo[0][0]*100:.1f}%, "
    f"P(Baja)={pred_prob_riesgo[0][1]*100:.1f}%"
)
print("Coeficientes [faltas, promedio]:", modelo_riesgo.coef_[0])
print("Intercepto:", modelo_riesgo.intercept_[0])