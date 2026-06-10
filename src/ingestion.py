import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(output_dir="data/raw"):
    """
    Genera datos simulados realistas para las variables agroclimáticas y socioeconómicas colombianas
    en caso de que no se cuente con archivos de datos reales.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    
    # 1. Definición de Municipios y Departamentos clave (ampliado a 30 municipios, 10 departamentos)
    locations = [
        # Antioquia
        {"departamento": "Antioquia", "municipio": "Fredonia", "lat": 5.92, "lon": -75.67, "altitud_m": 1800},
        {"departamento": "Antioquia", "municipio": "Jerico", "lat": 5.79, "lon": -75.78, "altitud_m": 1900},
        {"departamento": "Antioquia", "municipio": "Andes", "lat": 5.66, "lon": -75.87, "altitud_m": 1550},
        # Tolima
        {"departamento": "Tolima", "municipio": "Espinal", "lat": 4.15, "lon": -74.88, "altitud_m": 320},
        {"departamento": "Tolima", "municipio": "Ibague", "lat": 4.44, "lon": -75.24, "altitud_m": 1250},
        {"departamento": "Tolima", "municipio": "Purificacion", "lat": 3.86, "lon": -74.93, "altitud_m": 280},
        # Huila
        {"departamento": "Huila", "municipio": "Pitalito", "lat": 1.85, "lon": -76.05, "altitud_m": 1300},
        {"departamento": "Huila", "municipio": "Garzon", "lat": 2.20, "lon": -75.63, "altitud_m": 830},
        {"departamento": "Huila", "municipio": "Neiva", "lat": 2.93, "lon": -75.29, "altitud_m": 442},
        # Meta
        {"departamento": "Meta", "municipio": "Granada", "lat": 3.54, "lon": -73.70, "altitud_m": 370},
        {"departamento": "Meta", "municipio": "Villavicencio", "lat": 4.14, "lon": -73.63, "altitud_m": 460},
        {"departamento": "Meta", "municipio": "Acacias", "lat": 3.99, "lon": -73.76, "altitud_m": 498},
        # Cordoba
        {"departamento": "Cordoba", "municipio": "Cerete", "lat": 8.88, "lon": -75.79, "altitud_m": 15},
        {"departamento": "Cordoba", "municipio": "Monterias", "lat": 8.75, "lon": -75.88, "altitud_m": 18},
        {"departamento": "Cordoba", "municipio": "Sahagun", "lat": 8.95, "lon": -75.44, "altitud_m": 72},
        # Nariño
        {"departamento": "Narino", "municipio": "Pasto", "lat": 1.21, "lon": -77.28, "altitud_m": 2527},
        {"departamento": "Narino", "municipio": "Ipiales", "lat": 0.83, "lon": -77.64, "altitud_m": 2898},
        {"departamento": "Narino", "municipio": "Tumaco", "lat": 1.80, "lon": -78.77, "altitud_m": 5},
        # Boyacá
        {"departamento": "Boyaca", "municipio": "Tunja", "lat": 5.54, "lon": -73.36, "altitud_m": 2782},
        {"departamento": "Boyaca", "municipio": "Duitama", "lat": 5.82, "lon": -73.03, "altitud_m": 2530},
        {"departamento": "Boyaca", "municipio": "Sogamoso", "lat": 5.72, "lon": -72.93, "altitud_m": 2569},
        # Cundinamarca
        {"departamento": "Cundinamarca", "municipio": "Fusagasuga", "lat": 4.34, "lon": -74.36, "altitud_m": 1728},
        {"departamento": "Cundinamarca", "municipio": "Girardot", "lat": 4.30, "lon": -74.80, "altitud_m": 289},
        {"departamento": "Cundinamarca", "municipio": "Facatativa", "lat": 4.82, "lon": -74.36, "altitud_m": 2586},
        # Cauca
        {"departamento": "Cauca", "municipio": "Popayan", "lat": 2.44, "lon": -76.61, "altitud_m": 1738},
        {"departamento": "Cauca", "municipio": "Santander de Quilichao", "lat": 3.01, "lon": -76.49, "altitud_m": 1072},
        {"departamento": "Cauca", "municipio": "Miranda", "lat": 3.25, "lon": -76.23, "altitud_m": 990},
        # Cesar
        {"departamento": "Cesar", "municipio": "Valledupar", "lat": 10.46, "lon": -73.26, "altitud_m": 168},
        {"departamento": "Cesar", "municipio": "Aguachica", "lat": 8.31, "lon": -73.61, "altitud_m": 185},
        {"departamento": "Cesar", "municipio": "La Paz", "lat": 10.38, "lon": -73.17, "altitud_m": 202},
    ]
    
    crops = ["Cafe", "Cacao", "Arroz", "Maiz", "Platano"]
    
    print("Generando datos simulados en:", output_dir)
    
    # --- A. UPRA: Aptitud de Suelos ---
    soil_records = []
    for loc in locations:
        for crop in crops:
            # Lógica semi-realista: Tolima/Cordoba aptos para Arroz/Maíz, Antioquia/Huila para Café/Cacao
            suitability = "Alta"
            if crop in ["Cafe", "Cacao"] and loc["departamento"] in ["Tolima", "Cordoba"] and loc["municipio"] != "Ibague":
                suitability = "Baja" if np.random.rand() > 0.5 else "No Apta"
            elif crop in ["Arroz", "Maiz"] and loc["departamento"] in ["Antioquia", "Huila"]:
                suitability = "Media" if np.random.rand() > 0.5 else "Baja"
                
            soil_records.append({
                "departamento": loc["departamento"],
                "municipio": loc["municipio"],
                "latitud": loc["lat"],
                "longitud": loc["lon"],
                "cultivo": crop,
                "aptitud": suitability,
                "ph_suelo": round(np.random.uniform(5.2, 6.8), 2),
                "altitud_m": loc["altitud_m"],
                "materia_organica_pct": round(np.random.uniform(1.5, 5.0), 2),
                "pendiente_pct": round(np.random.uniform(1.0, 35.0) if loc["departamento"] in ["Antioquia", "Huila", "Tolima"] else np.random.uniform(0.5, 5.0), 2),
                "textura": np.random.choice(["Franco-Arenoso", "Franco-Arcilloso", "Franco", "Arcilloso"])
            })
    df_soils = pd.DataFrame(soil_records)
    df_soils.to_csv(os.path.join(output_dir, "upra_aptitud_suelos.csv"), index=False, encoding='utf-8')
    
    # --- B. IDEAM: Clima Histórico ---
    # Registros diarios de los últimos 3 años
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    
    weather_records = []
    for loc in locations:
        for date in date_range:
            # Variaciones estacionales aproximadas para Colombia (épocas de lluvia Abr-May y Oct-Nov)
            month = date.month
            is_rainy_season = month in [4, 5, 10, 11]
            
            base_temp = 22.0 if loc["departamento"] in ["Antioquia", "Huila"] else 28.0
            temp = base_temp + np.random.normal(0, 1.5)
            
            base_rain = 8.0 if is_rainy_season else 2.0
            rain = max(0.0, base_rain + np.random.normal(0, 5.0)) if np.random.rand() > (0.3 if is_rainy_season else 0.7) else 0.0
            
            humidity = 70.0 + (15.0 if is_rainy_season else 0.0) + np.random.normal(0, 5.0)
            humidity = min(100.0, max(40.0, humidity))
            
            weather_records.append({
                "fecha": date.strftime("%Y-%m-%d"),
                "departamento": loc["departamento"],
                "municipio": loc["municipio"],
                "temperatura_promedio_c": round(temp, 1),
                "precipitacion_diaria_mm": round(rain, 1),
                "humedad_relativa_pct": round(humidity, 1),
                "brillo_solar_horas": round(max(0.0, 8.0 - (rain * 0.4) + np.random.normal(0, 1.0)), 1)
            })
    df_weather = pd.DataFrame(weather_records)
    df_weather.to_csv(os.path.join(output_dir, "ideam_clima_historico.csv"), index=False, encoding='utf-8')
    
    # --- C. FINAGRO: Créditos Financieros Sectoriales ---
    finagro_records = []
    years = [2023, 2024, 2025]
    for loc in locations:
        for year in years:
            for crop in crops:
                finagro_records.append({
                    "anio": year,
                    "departamento": loc["departamento"],
                    "municipio": loc["municipio"],
                    "cultivo": crop,
                    "monto_creditos_cop": int(np.random.uniform(50, 800) * 1e6),
                    "numero_creditos_aprobados": int(np.random.randint(5, 120)),
                    "tasa_interes_promedio": round(np.random.uniform(10.5, 16.0), 2)
                })
    df_finagro = pd.DataFrame(finagro_records)
    df_finagro.to_csv(os.path.join(output_dir, "finagro_creditos.csv"), index=False, encoding='utf-8')
    
    # --- D. SIPSA / DANE: Precios de Mercado Históricos ---
    price_records = []
    price_dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="W") # Precios semanales
    base_prices = {"Cafe": 12000, "Cacao": 9500, "Arroz": 2500, "Maiz": 1800, "Platano": 1500}
    
    for date in price_dates:
        for crop in crops:
            # Simulación de variación de precios con tendencia e inflación
            trend = (date - price_dates[0]).days * 0.5
            noise = np.random.normal(0, base_prices[crop] * 0.05)
            price = base_prices[crop] + trend + noise
            price_records.append({
                "fecha": date.strftime("%Y-%m-%d"),
                "cultivo": crop,
                "precio_kg_promedio_cop": round(max(500, price), 0),
                "central_abasto": np.random.choice(["Central Mayorista de Antioquia", "Corabastos Bogota", "Cavasa Cali"])
            })
    df_prices = pd.DataFrame(price_records)
    df_prices.to_csv(os.path.join(output_dir, "sipsa_precios_mercado.csv"), index=False, encoding='utf-8')
    
    # --- E. Seguridad Jurídica de Tierras (UPRA / ANT - Agencia Nacional de Tierras) ---
    legal_records = []
    # Factores por departamento: zonas históricamente afectadas por conflicto tienen menor ISJ
    conflict_dept_factor = {
        "Antioquia": 0.75, "Cordoba": 0.60, "Meta": 0.65, "Cauca": 0.55,
        "Narino": 0.58, "Cesar": 0.70, "Tolima": 0.72, "Huila": 0.80,
        "Boyaca": 0.90, "Cundinamarca": 0.88
    }
    for loc in locations:
        dept = loc["departamento"]
        cf = conflict_dept_factor.get(dept, 0.75)
        ha_catastradas = round(np.random.uniform(8000, 120000), 1)
        ha_formalizadas = round(ha_catastradas * cf * np.random.uniform(0.85, 1.0), 1)
        legal_records.append({
            "departamento": dept,
            "municipio": loc["municipio"],
            "hectareas_catastradas": ha_catastradas,
            "hectareas_formalizadas": ha_formalizadas,
            "predios_formalizados": int(ha_formalizadas / np.random.uniform(5, 35)),
            "solicitudes_restitucion_activas": int(np.random.randint(0, 80) * (1.5 - cf)),
            "conflicto_uso_suelo_pct": round((1.0 - cf) * np.random.uniform(60, 100), 1),
            "indice_seguridad_juridica": round(cf * 100 * np.random.uniform(0.90, 1.05), 1)  # Porcentaje de formalización
        })
    df_legal = pd.DataFrame(legal_records)
    # Asegurar que el índice no supere 100
    df_legal["indice_seguridad_juridica"] = df_legal["indice_seguridad_juridica"].clip(upper=100.0)
    df_legal.to_csv(os.path.join(output_dir, "seguridad_juridica_tierras.csv"), index=False, encoding='utf-8')
    
    # --- F. Producción Histórica de Cosechas (Etiqueta para entrenar IA) ---
    production_records = []
    for loc in locations:
        for year in years:
            for crop in crops:
                # El rendimiento depende de la aptitud del suelo de este municipio y el clima del año
                soil_apt = df_soils[(df_soils["municipio"] == loc["municipio"]) & (df_soils["cultivo"] == crop)]["aptitud"].values
                apt_factor = {"Alta": 1.2, "Media": 1.0, "Baja": 0.7, "No Apta": 0.3}
                factor = apt_factor.get(soil_apt[0], 0.5) if len(soil_apt) > 0 else 0.8
                
                # Promedios del clima de este año en este municipio
                yearly_weather = df_weather[
                    (df_weather["municipio"] == loc["municipio"]) & 
                    (pd.to_datetime(df_weather["fecha"]).dt.year == year)
                ]
                mean_rain = yearly_weather["precipitacion_diaria_mm"].mean() * 365 if len(yearly_weather) > 0 else 1800.0
                
                # Clima óptimo: lluvias moderadas (1200 - 2200 mm/año)
                climate_factor = 1.0
                if mean_rain < 1000 or mean_rain > 2500:
                    climate_factor = 0.75
                
                base_yield = {"Cafe": 1.4, "Cacao": 0.6, "Arroz": 5.5, "Maiz": 3.8, "Platano": 12.0} # Toneladas / Hectárea
                predicted_yield = base_yield[crop] * factor * climate_factor * np.random.uniform(0.9, 1.1)
                
                area_planted = np.random.uniform(500, 8000)
                production = area_planted * predicted_yield
                
                production_records.append({
                    "anio": year,
                    "departamento": loc["departamento"],
                    "municipio": loc["municipio"],
                    "cultivo": crop,
                    "hectareas_sembradas": round(area_planted, 1),
                    "produccion_obtenida_ton": round(production, 1),
                    "rendimiento_ton_ha": round(predicted_yield, 2)
                })
    df_production = pd.DataFrame(production_records)
    df_production.to_csv(os.path.join(output_dir, "produccion_historica.csv"), index=False, encoding='utf-8')
    
    # --- G. ICA: Inventario Ganadero Nacional (Bovinos y Porcinos) ---
    livestock_records = []
    for loc in locations:
        for year in years:
            pob_bovina = int(np.random.uniform(5000, 75000))
            pob_porcina = int(np.random.uniform(2000, 35000))
            prod_leche = round(pob_bovina * np.random.uniform(2.5, 4.5), 1)
            vacunacion = round(np.random.uniform(92.0, 99.8), 2)
            
            livestock_records.append({
                "anio": year,
                "departamento": loc["departamento"],
                "municipio": loc["municipio"],
                "poblacion_bovina": pob_bovina,
                "poblacion_porcina": pob_porcina,
                "produccion_leche_litros_dia": prod_leche,
                "vacunacion_fiebre_aftosa_pct": vacunacion
            })
    df_livestock = pd.DataFrame(livestock_records)
    df_livestock.to_csv(os.path.join(output_dir, "inventario_ganadero_nacional.csv"), index=False, encoding='utf-8')
    
    print("Datos simulados creados con éxito.")

def load_challenge_metadata(workspace_dir="."):
    """
    Carga e integra TODOS los archivos CSV del concurso Datos al Ecosistema 2026
    (catálogo principal + coberturas local, nacional, regional + entidades alcaldía y ministerio)
    en un único catálogo unificado y deduplicado por UID.
    """
    frames = []
    
    # Patrón de archivos del concurso (todos los reto_04_agricultura_*.csv)
    if os.path.exists(workspace_dir):
        for file in sorted(os.listdir(workspace_dir)):
            if file.startswith("reto_04_agricultura_") and file.endswith(".csv"):
                fpath = os.path.join(workspace_dir, file)
                try:
                    df_part = pd.read_csv(fpath, encoding='utf-8')
                    if not df_part.empty:
                        frames.append(df_part)
                        print(f"  [OK] Cargado: {file} ({df_part.shape[0]} filas)")
                except Exception as e:
                    print(f"  [ERR] Error al leer {file}: {e}")
    
    if not frames:
        print("Catálogo de retos de agricultura no encontrado en el directorio raíz.")
        return pd.DataFrame()
    
    df_combined = pd.concat(frames, ignore_index=True)
    
    # Deduplicar por UID (priorizar el registro con más columnas rellenas)
    if "UID" in df_combined.columns:
        df_combined = (
            df_combined
            .sort_values(by=df_combined.columns.tolist(), na_position='last')
            .drop_duplicates(subset=["UID"], keep='first')
            .reset_index(drop=True)
        )
    
    print(f"\n[OK] Catalogo unificado: {df_combined.shape[0]} datasets ({len(frames)} archivos fusionados)")
    return df_combined

def load_local_dataset(filename, data_dir="data/raw"):
    """
    Carga un archivo CSV local. Si no existe, genera los datos simulados primero.
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        generate_mock_data(data_dir)
    return pd.read_csv(path, encoding='utf-8')

if __name__ == "__main__":
    generate_mock_data()
