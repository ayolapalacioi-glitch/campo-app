import os
import pandas as pd
import numpy as np
from src.ingestion import load_local_dataset, load_challenge_metadata

# Diccionario de geocodificación para departamentos y municipios de Colombia
DEPARTAMENTO_COORDINATES = {
    "boyaca": (5.53, -73.36),
    "boyacá": (5.53, -73.36),
    "casanare": (5.35, -72.40),
    "caldas": (5.07, -75.52),
    "cauca": (2.44, -76.61),
    "guaviare": (2.56, -72.64),
    "huila": (2.92, -75.28),
    "valle del cauca": (3.45, -76.53),
    "valle": (3.45, -76.53),
    "norte de santander": (7.89, -72.50),
    "risaralda": (4.81, -75.69),
    "santander": (7.12, -73.12),
    "sucre": (9.30, -75.39),
    "meta": (4.14, -73.63),
    "cordoba": (8.75, -75.88),
    "córdoba": (8.75, -75.88),
    "cundinamarca": (4.71, -74.19),
    "tolima": (4.44, -75.24),
    "antioquia": (6.25, -75.56),
    "atlantico": (10.96, -74.79),
    "atlántico": (10.96, -74.79),
    "quindio": (4.53, -75.67),
    "quindío": (4.53, -75.67),
    "san andres": (12.58, -81.70),
    "san andrés": (12.58, -81.70),
    "bogota": (4.61, -74.08),
    "bogotá": (4.61, -74.08),
    "bogota d.c.": (4.61, -74.08),
    "bogotá d.c.": (4.61, -74.08),
    "nacional": (4.61, -74.08),
}

def run_diagnostic(df, name="Dataset"):
    """
    Imprime un diagnóstico rápido de la calidad del dataset.
    """
    print(f"\n--- DIAGNÓSTICO PARA: {name} ---")
    print(f"Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
    print("Valores nulos por columna:")
    nulls = df.isnull().sum()
    for col, count in nulls.items():
        if count > 0:
            print(f"  - {col}: {count} ({round(count/len(df)*100, 2)}%)")
        else:
            print(f"  - {col}: 0 (Limpio)")
    print(f"Filas duplicadas: {df.duplicated().sum()}")

def geocode_department(dept_name):
    """
    Retorna la latitud y longitud de un departamento de Colombia.
    """
    if pd.isna(dept_name):
        return 4.61, -74.08 # Bogotá/Nacional por defecto
    
    name_norm = str(dept_name).strip().lower()
    # Buscar coincidencia exacta o por subcadena
    for key, coords in DEPARTAMENTO_COORDINATES.items():
        if key in name_norm or name_norm in key:
            return coords
            
    return 4.61, -74.08 # Bogotá/Nacional por defecto

def clean_challenge_catalog(workspace_dir=".", processed_dir="data/processed"):
    """
    Carga, limpia, georreferencia en Colombia y guarda el catálogo de retos.
    """
    df_catalog = load_challenge_metadata(workspace_dir)
    if df_catalog.empty:
        print("El catálogo está vacío o no se encontró.")
        return df_catalog
        
    run_diagnostic(df_catalog, "Catálogo de Retos Crudo")
    
    # 1. Definir columnas útiles
    cols_to_keep = [
        "UID", "Titulo", "Descripción", "Número de Filas", "Número de Columnas",
        "alcance_geografico", "es_viable", "ds_encabezados_utiles",
        "Información de la Entidad: Nombre de la Entidad", "Información de la Entidad: Sector",
        "Información de la Entidad: Orden", "Información de la Entidad: Departamento",
        "Información de la Entidad: Municipio", "Categoría", "Etiqueta",
        "Fecha de última actualización de datos (UTC)", "API", "url", "ds_justificacion",
        "ds_score_relevancia",
        # Columnas adicionales del concurso (Datos al Ecosistema 2026)
        "ds_calidad_datos", "ds_reto_principal", "ds_retos_secundarios", "ds_potencial_integracion",
        "ds_reto_sugerido", "ds_justificacion_sugerida"
    ]
    
    existing_cols = [c for c in cols_to_keep if c in df_catalog.columns]
    df_clean = df_catalog[existing_cols].copy()
    
    # 2. Limpieza de tipos y nulos
    df_clean["es_viable"] = df_clean["es_viable"].fillna(True).astype(bool)
    
    # Limpiar columnas numéricas
    for col in ["Número de Filas", "Número de Columnas"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace(".", "", regex=False)
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
            
    if "ds_score_relevancia" in df_clean.columns:
        df_clean["ds_score_relevancia"] = pd.to_numeric(df_clean["ds_score_relevancia"], errors='coerce').fillna(3.0)
    
    if "ds_calidad_datos" in df_clean.columns:
        df_clean["ds_calidad_datos"] = pd.to_numeric(df_clean["ds_calidad_datos"], errors='coerce').fillna(2.0)

    if "ds_potencial_integracion" in df_clean.columns:
        df_clean["ds_potencial_integracion"] = df_clean["ds_potencial_integracion"].map(
            lambda x: True if str(x).strip().lower() in ['true', '1', 'yes', 'si', 'sí'] else False
            if str(x).strip().lower() in ['false', '0', 'no'] else None
        ).fillna(False)
        
    # Rellenar textos nulos
    text_cols = df_clean.select_dtypes(include=['object']).columns
    df_clean[text_cols] = df_clean[text_cols].fillna("No disponible")
    
    # 3. Aplicar Geocodificación real de Colombia
    lats = []
    lons = []
    
    dept_col = "Información de la Entidad: Departamento"
    mun_col = "Información de la Entidad: Municipio"
    
    for idx, row in df_clean.iterrows():
        dept = row[dept_col] if dept_col in row else "Nacional"
        mun = row[mun_col] if mun_col in row else "No disponible"
        
        # Primero intentar geocodificar con el municipio (ej. Bogotá)
        if mun != "No disponible" and str(mun).strip().lower() in DEPARTAMENTO_COORDINATES:
            lat, lon = DEPARTAMENTO_COORDINATES[str(mun).strip().lower()]
        else:
            lat, lon = geocode_department(dept)
            
        lats.append(lat)
        lons.append(lon)
        
    df_clean["latitud"] = lats
    df_clean["longitud"] = lons
    
    # Guardar
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "cleaned_datasets_catalog.csv")
    df_clean.to_csv(out_path, index=False, encoding='utf-8')
    print(f"Catálogo de retos georreferenciado guardado en: {out_path}")
    
    # Copia para Power BI
    pbi_dir = os.path.join(processed_dir, "power_bi")
    os.makedirs(pbi_dir, exist_ok=True)
    df_clean.to_csv(os.path.join(pbi_dir, "catalog_datasets.csv"), index=False, encoding='utf-8')
    
    return df_clean

def process_and_consolidate(raw_dir="data/raw", processed_dir="data/processed"):
    """
    Carga los datos agrícolas, los limpia y exporta. Llama también a la limpieza del catálogo.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    df_soils = load_local_dataset("upra_aptitud_suelos.csv", raw_dir)
    df_weather = load_local_dataset("ideam_clima_historico.csv", raw_dir)
    df_finagro = load_local_dataset("finagro_creditos.csv", raw_dir)
    df_prices = load_local_dataset("sipsa_precios_mercado.csv", raw_dir)
    df_legal = load_local_dataset("seguridad_juridica_tierras.csv", raw_dir)
    df_production = load_local_dataset("produccion_historica.csv", raw_dir)
    df_livestock = load_local_dataset("inventario_ganadero_nacional.csv", raw_dir)
    
    df_weather["fecha"] = pd.to_datetime(df_weather["fecha"])
    df_weather["anio"] = df_weather["fecha"].dt.year
    
    for col in ["temperatura_promedio_c", "precipitacion_diaria_mm", "humedad_relativa_pct", "brillo_solar_horas"]:
        df_weather[col] = df_weather.groupby(["departamento", "municipio"])[col].transform(lambda x: x.interpolate(method='linear').bfill().ffill())
        
    df_weather_yearly = df_weather.groupby(["anio", "departamento", "municipio"]).agg(
        temp_media_c=("temperatura_promedio_c", "mean"),
        precipitacion_anual_mm=("precipitacion_diaria_mm", "sum"),
        humedad_media_pct=("humedad_relativa_pct", "mean"),
        brillo_solar_anual_horas=("brillo_solar_horas", "sum")
    ).reset_index()
    
    df_prices["fecha"] = pd.to_datetime(df_prices["fecha"])
    df_prices["anio"] = df_prices["fecha"].dt.year
    df_prices_yearly = df_prices.groupby(["anio", "cultivo"])["precio_kg_promedio_cop"].mean().reset_index()
    df_prices_yearly.rename(columns={"precio_kg_promedio_cop": "precio_kg_anual_promedio_cop"}, inplace=True)
    
    master_df = df_production.copy()
    
    master_df = pd.merge(
        master_df, 
        df_soils[["departamento", "municipio", "cultivo", "aptitud", "ph_suelo", "altitud_m", "materia_organica_pct", "pendiente_pct", "textura", "latitud", "longitud"]],
        on=["departamento", "municipio", "cultivo"],
        how="left"
    )
    
    master_df = pd.merge(
        master_df,
        df_weather_yearly,
        on=["anio", "departamento", "municipio"],
        how="left"
    )
    
    master_df = pd.merge(
        master_df,
        df_finagro[["anio", "departamento", "municipio", "cultivo", "monto_creditos_cop", "numero_creditos_aprobados", "tasa_interes_promedio"]],
        on=["anio", "departamento", "municipio", "cultivo"],
        how="left"
    )
    
    master_df = pd.merge(
        master_df,
        df_prices_yearly,
        on=["anio", "cultivo"],
        how="left"
    )
    
    master_df = pd.merge(
        master_df,
        df_legal[["departamento", "municipio", "hectareas_formalizadas", "solicitudes_restitucion_activas", "indice_seguridad_juridica"]],
        on=["departamento", "municipio"],
        how="left"
    )
    
    master_df = pd.merge(
        master_df,
        df_livestock[["anio", "departamento", "municipio", "poblacion_bovina", "poblacion_porcina", "produccion_leche_litros_dia", "vacunacion_fiebre_aftosa_pct"]],
        on=["anio", "departamento", "municipio"],
        how="left"
    )
    
    numeric_cols = master_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        master_df[col] = master_df[col].fillna(master_df.groupby("cultivo")[col].transform("mean"))
        
    master_df["textura"] = master_df["textura"].fillna("Franco")
    master_df["aptitud"] = master_df["aptitud"].fillna("Media")
    
    # Guardar en local
    output_path = os.path.join(processed_dir, "consolidated_data.csv")
    master_df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Exportar para Power BI
    pbi_dir = os.path.join(processed_dir, "power_bi")
    os.makedirs(pbi_dir, exist_ok=True)
    master_df.to_csv(os.path.join(pbi_dir, "agromaster_data.csv"), index=False, encoding='utf-8')
    df_weather_yearly.to_csv(os.path.join(pbi_dir, "clima_ideam_anual.csv"), index=False, encoding='utf-8')
    df_prices_yearly.to_csv(os.path.join(pbi_dir, "precios_sipsa_anual.csv"), index=False, encoding='utf-8')
    df_livestock.to_csv(os.path.join(pbi_dir, "inventario_ganadero_anual.csv"), index=False, encoding='utf-8')
    
    # Procesar catálogo de retos
    clean_challenge_catalog(processed_dir=processed_dir)
    
    return master_df

if __name__ == "__main__":
    process_and_consolidate()
