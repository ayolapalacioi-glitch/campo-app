import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Intentar importar PySpark para soporte Big Data
try:
    from pyspark.sql import SparkSession
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False
    print("PySpark no está instalado en este entorno local. Se usará Pandas como fallback.")

# 1. INICIALIZACIÓN DE ENTORNO BIG DATA (PySpark para datos del IDEAM)
if HAS_PYSPARK:
    spark = SparkSession.builder.appName("Antigravity_Agro_AI").getOrCreate()
else:
    spark = None

# 2. INGESTA Y LIMPIEZA DE DATOS (Data Preparation)
def load_and_prep_data():
    print("Cargando datos de UPRA, IDEAM, ICA y FINAGRO...")
    
    # Carga de datos procesados (Master Data unificado por municipio)
    if os.path.exists("data/processed/consolidated_data.csv"):
        df_master = pd.read_csv("data/processed/consolidated_data.csv")
    else:
        # Fallback de simulación en caso de no existir
        print("Dataset consolidado no encontrado. Ejecutando pipeline de ingesta preliminar...")
        from src.ingestion import generate_mock_data
        from src.cleaning import process_and_consolidate
        generate_mock_data("data/raw")
        process_and_consolidate("data/raw", "data/processed")
        df_master = pd.read_csv("data/processed/consolidated_data.csv")
        
    return df_master

# 3. MODELAMIENTO PREDICTIVO (Modeling)
def train_hybrid_model(df):
    print("\nEntrenando arquitectura híbrida de Machine Learning...")
    
    # Definir características predictoras
    categorical_features = ["cultivo", "textura"]
    numeric_features = ["ph_suelo", "altitud_m", "pendiente_pct", "materia_organica_pct", "temp_media_c", "precipitacion_anual_mm"]
    
    # Codificar variables categóricas para XGBoost
    df_encoded = pd.get_dummies(df[categorical_features + numeric_features + ["rendimiento_ton_ha"]], columns=categorical_features, drop_first=True)
    
    X = df_encoded.drop(columns=["rendimiento_ton_ha"])
    y = df_encoded["rendimiento_ton_ha"].fillna(df_encoded["rendimiento_ton_ha"].mean())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelo 1: Random Forest (Robusto para datos agrícolas tabulares)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Modelo 2: XGBoost (Excelente para capturar patrones de riesgo climático extremo)
    xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    return rf_model, xgb_model, X_test, y_test

# 4. EVALUACIÓN (Evaluation)
def evaluate_models(rf_model, xgb_model, X_test, y_test):
    rf_preds = rf_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)
    
    rmse_rf = mean_squared_error(y_test, rf_preds) ** 0.5
    rmse_xgb = mean_squared_error(y_test, xgb_preds) ** 0.5
    
    print("\n--- EVALUACIÓN DE DESEMPEÑO DEL MODELO HÍBRIDO ---")
    print(f"Error RMSE (Random Forest): {round(rmse_rf, 4)} Ton/Ha")
    print(f"Error RMSE (XGBoost): {round(rmse_xgb, 4)} Ton/Ha")
    
    # Exportar el modelo de producción (.joblib)
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(xgb_model, "data/processed/crop_yield_model_xgb.joblib")
    print("Modelo XGBoost exportado exitosamente a: data/processed/crop_yield_model_xgb.joblib")

if __name__ == "__main__":
    datos_limpios = load_and_prep_data()
    modelo_rf, modelo_xgb, X_test, y_test = train_hybrid_model(datos_limpios)
    evaluate_models(modelo_rf, modelo_xgb, X_test, y_test)
    print("\n[ÉXITO] Pipeline del modelo ejecutado correctamente.")
