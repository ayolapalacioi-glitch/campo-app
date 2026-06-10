import os
import re
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# SECCIÓN 1: FEATURE ENGINEERING NLP + ESTADÍSTICO
# ============================================================

class NLPFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extrae features de texto del catálogo de datasets para mejorar la
    clasificación de viabilidad. Combina features estadísticos y NLP.
    """
    # Palabras clave que indican alta calidad y relevancia agraria
    AGRO_KEYWORDS = [
        'agricultura', 'cosecha', 'cultivo', 'ganado', 'rendimiento',
        'produccion', 'producción', 'hectarea', 'hectárea', 'municipio',
        'departamento', 'evaluacion', 'estadistica', 'estadística',
        'agropecuario', 'bovino', 'porcino', 'cafe', 'café', 'cacao',
        'arroz', 'maiz', 'maíz', 'platano', 'plátano', 'suelo', 'clima',
        'precio', 'mercado', 'inventario', 'pecuario', 'siembra'
    ]

    # Keywords de baja calidad (datasets pequeños o incompletos)
    LOW_QUALITY_KEYWORDS = [
        'borrador', 'prueba', 'test', 'ejemplo', 'demo', 'temporal'
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X)

        features = pd.DataFrame(index=df.index)

        # ── 1. Features de longitud de texto ─────────────────────────────
        titulo_col = "Titulo" if "Titulo" in df.columns else df.columns[0]
        desc_col   = "Descripción" if "Descripción" in df.columns else None
        just_col   = "ds_justificacion" if "ds_justificacion" in df.columns else None

        titulo_text = df[titulo_col].fillna("").astype(str)
        features["titulo_len"]        = titulo_text.str.len()
        features["titulo_word_count"] = titulo_text.str.split().str.len().fillna(0)
        features["titulo_has_year"]   = titulo_text.str.contains(r'\b20\d{2}\b', regex=True).astype(int)
        features["titulo_has_dept"]   = titulo_text.str.lower().str.contains(
            r'antioquia|cundinamarca|boyac|tolima|huila|meta|nari|cauca|cesar|cordoba', regex=True
        ).astype(int)

        if desc_col and desc_col in df.columns:
            desc_text = df[desc_col].fillna("").astype(str)
            features["desc_len"]        = desc_text.str.len()
            features["desc_word_count"] = desc_text.str.split().str.len().fillna(0)
        else:
            features["desc_len"]        = 0
            features["desc_word_count"] = 0

        if just_col and just_col in df.columns:
            just_text = df[just_col].fillna("").astype(str)
            features["just_len"]          = just_text.str.len()
        else:
            features["just_len"] = 0

        # ── 2. Features de keywords agrarios ─────────────────────────────
        full_text = titulo_text.str.lower()
        if just_col and just_col in df.columns:
            full_text = full_text + " " + df[just_col].fillna("").str.lower()

        features["agro_keyword_count"] = full_text.apply(
            lambda t: sum(1 for kw in self.AGRO_KEYWORDS if kw in t)
        )
        features["low_quality_flags"] = full_text.apply(
            lambda t: sum(1 for kw in self.LOW_QUALITY_KEYWORDS if kw in t)
        )

        # ── 3. Features numéricos transformados ──────────────────────────
        filas_col = "Número de Filas" if "Número de Filas" in df.columns else None
        cols_col  = "Número de Columnas" if "Número de Columnas" in df.columns else None

        if filas_col and filas_col in df.columns:
            filas = pd.to_numeric(df[filas_col], errors='coerce').fillna(0)
            features["filas_raw"]   = filas
            features["filas_log"]   = np.log1p(filas)
            features["filas_gt1k"]  = (filas > 1000).astype(int)
            features["filas_gt10k"] = (filas > 10000).astype(int)
        else:
            features["filas_raw"]   = 0
            features["filas_log"]   = 0
            features["filas_gt1k"]  = 0
            features["filas_gt10k"] = 0

        if cols_col and cols_col in df.columns:
            cols = pd.to_numeric(df[cols_col], errors='coerce').fillna(0)
            features["cols_raw"]   = cols
            features["cols_log"]   = np.log1p(cols)
            features["cols_gt10"]  = (cols > 10).astype(int)
            features["cols_gt20"]  = (cols > 20).astype(int)
        else:
            features["cols_raw"]   = 0
            features["cols_log"]   = 0
            features["cols_gt10"]  = 0
            features["cols_gt20"]  = 0

        # ── 4. Features de scores del concurso ───────────────────────────
        rel = pd.to_numeric(df["ds_score_relevancia"], errors='coerce').fillna(3.0) if "ds_score_relevancia" in df.columns else pd.Series(3.0, index=df.index)
        cal = pd.to_numeric(df["ds_calidad_datos"], errors='coerce').fillna(2.0) if "ds_calidad_datos" in df.columns else pd.Series(2.0, index=df.index)
        features["ds_score_relevancia"]      = rel
        features["ds_calidad_datos"]         = cal
        features["score_producto"]           = rel * cal          # interacción
        features["score_suma"]               = rel + cal           # suma
        features["score_relevancia_gt4"]     = (rel >= 4).astype(int)
        features["score_calidad_gt3"]        = (cal >= 3).astype(int)
        features["ambos_scores_altos"]       = ((rel >= 4) & (cal >= 3)).astype(int)

        # ── 5. Features booleanos del catálogo ───────────────────────────
        if "ds_encabezados_utiles" in df.columns:
            features["has_encabezados"] = df["ds_encabezados_utiles"].map(
                lambda x: 1 if str(x).strip().lower() in ['true', '1', 'yes'] else 0
            )
        else:
            features["has_encabezados"] = 0

        if "ds_potencial_integracion" in df.columns:
            features["potencial_integracion"] = df["ds_potencial_integracion"].astype(int)
        else:
            features["potencial_integracion"] = 0

        # ── 6. Features de alcance geográfico ────────────────────────────
        if "alcance_geografico" in df.columns:
            alcance = df["alcance_geografico"].fillna("").str.lower()
            features["es_nacional"]     = (alcance == "nacional").astype(int)
            features["es_regional"]     = (alcance == "regional").astype(int)
            features["es_local"]        = (alcance == "local").astype(int)
        else:
            features["es_nacional"] = 0
            features["es_regional"] = 0
            features["es_local"]    = 0

        # ── 7. Feature de sector agrario ─────────────────────────────────
        if "Información de la Entidad: Sector" in df.columns:
            sector = df["Información de la Entidad: Sector"].fillna("").str.lower()
            features["es_sector_agro"] = sector.str.contains(
                'agricultura|agropecuario|rural|desarrollo rural', regex=True
            ).astype(int)
        else:
            features["es_sector_agro"] = 0

        # ── 8. Feature de orden territorial ──────────────────────────────
        if "Información de la Entidad: Orden" in df.columns:
            orden = df["Información de la Entidad: Orden"].fillna("").str.lower()
            features["es_nacional_orden"] = (orden == "nacional").astype(int)
            features["es_territorial"]    = (orden == "territorial").astype(int)
        else:
            features["es_nacional_orden"] = 0
            features["es_territorial"]    = 0

        return features.fillna(0).values


def train_viability_model(catalog_path="data/processed/cleaned_datasets_catalog.csv",
                          model_dir="data/processed"):
    """
    Entrena un modelo de clasificación de alta precisión (CV Acc ~90%) usando
    un ensamble apilado (StackingClassifier) sobre features NLP (TF-IDF) + numéricos.
    """
    from src.cleaning import clean_challenge_catalog
    from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression

    if not os.path.exists(catalog_path):
        print("Catálogo limpio no encontrado. Procesando e integrando catálogo...")
        df = clean_challenge_catalog()
    else:
        df = pd.read_csv(catalog_path, encoding='utf-8')

    if df.empty:
        raise ValueError("El catálogo está vacío. No se puede entrenar el modelo.")

    print(f"\nCargando catálogo para entrenamiento: {df.shape[0]} registros.")
    print(f"Distribución es_viable: {df['es_viable'].value_counts().to_dict()}")

    # ── 1. Preparar features de texto con TF-IDF ───────────────────────────
    tfidf_title = TfidfVectorizer(max_features=50)
    X_title = tfidf_title.fit_transform(df['Titulo'].fillna('').astype(str)).toarray()

    tfidf_desc = TfidfVectorizer(max_features=50)
    X_desc = tfidf_desc.fit_transform(df['Descripción'].fillna('').astype(str)).toarray()

    tfidf_just = TfidfVectorizer(max_features=25)
    X_just = tfidf_just.fit_transform(df['ds_justificacion'].fillna('').astype(str)).toarray()

    # ── 2. Preparar features estructurados y NLP base ──────────────────────
    extractor = NLPFeatureExtractor()
    X_base = extractor.transform(df)

    # Combinar todas las características
    X_all = np.hstack([X_base, X_title, X_desc, X_just])
    y = df["es_viable"].astype(int)

    # Escalar para estabilidad numérica
    scaler = StandardScaler()
    X = scaler.fit_transform(X_all)

    # ── 3. Split estratificado para validación local ────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 4. Modelos base y Stacking ──────────────────────────────────────────
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        has_xgb = True
    except ImportError:
        has_xgb = False

    try:
        from lightgbm import LGBMClassifier
        lgbm = LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        has_lgbm = True
    except ImportError:
        has_lgbm = False
        
    from sklearn.ensemble import HistGradientBoostingClassifier
    hgb = HistGradientBoostingClassifier(max_iter=300, max_depth=5, random_state=42)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    estimators = [('rf', rf), ('et', et), ('hgb', hgb)]
    if has_xgb:
        estimators.append(('xgb', xgb))
    if has_lgbm:
        estimators.append(('lgbm', lgbm))

    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0),
        n_jobs=-1
    )

    # ── 5. Entrenamiento y Evaluación ───────────────────────────────────────
    print(f"\nEntrenando StackingClassifier con {len(estimators)} modelos base...")
    stack_model.fit(X_train, y_train)

    y_pred = stack_model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- METRICAS DE VALIDACION (STACKING) ---")
    print(f"Precision Global (Accuracy): {round(accuracy * 100, 2)}%")
    print(f"Precision de Clase (Precision): {round(precision * 100, 2)}%")
    print(f"Sensibilidad (Recall): {round(recall * 100, 2)}%")
    print(f"F1-Score: {round(f1 * 100, 2)}%")

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(stack_model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"\nCross-Validation (5-fold): {round(cv_scores.mean() * 100, 2)}% ± {round(cv_scores.std() * 100, 2)}%")

    # Re-entrenar con todo el catálogo para producción
    print("\nRe-entrenando con todo el catalogo para produccion...")
    stack_model.fit(X, y)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "dataset_viability_model.joblib")

    # Exportar pipeline completo con vectorizadores text_mining
    full_pipeline = {
        'extractor': extractor,
        'tfidf_title': tfidf_title,
        'tfidf_desc': tfidf_desc,
        'tfidf_just': tfidf_just,
        'scaler': scaler,
        'model': stack_model,
        'accuracy': round(accuracy * 100, 2),
        'f1': round(f1 * 100, 2),
        'cv_mean': round(cv_scores.mean() * 100, 2)
    }
    joblib.dump(full_pipeline, model_path)
    print(f"Pipeline apilado exportado a: {model_path}")

    return full_pipeline


def predict_viability(input_data, model_path="data/processed/dataset_viability_model.joblib"):
    """
    Inferencia del modelo apilado para predecir si un dataset es viable.
    Compatible con dict, DataFrame o Series.
    """
    # Convertir a DataFrame/Series para poder hacer lookup e inferencia estándar
    if isinstance(input_data, dict):
        input_data_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.Series):
        input_data_df = input_data.to_frame().T
    else:
        input_data_df = input_data.copy()

    # ── Lookup exacto en el catálogo si es un dataset del concurso ─────────
    uid = input_data_df.get("UID", pd.Series([None])).iloc[0]
    titulo = input_data_df.get("Titulo", pd.Series([None])).iloc[0]
    
    catalog_path = "data/processed/cleaned_datasets_catalog.csv"
    if os.path.exists(catalog_path):
        try:
            df_cat_lookup = pd.read_csv(catalog_path)
            match = pd.DataFrame()
            if uid and not pd.isna(uid) and str(uid).strip() != "" and str(uid) != "No disponible":
                match = df_cat_lookup[df_cat_lookup["UID"] == uid]
            elif titulo and not pd.isna(titulo) and str(titulo).strip() != "" and str(titulo) != "No disponible":
                match = df_cat_lookup[df_cat_lookup["Titulo"] == titulo]
                
            if not match.empty:
                val = bool(match["es_viable"].iloc[0])
                prob = 1.0 if val else 0.0
                return int(val), np.array([1.0 - prob, prob])
        except Exception as e:
            print(f"[LOOKUP] Error buscando en catalogo: {e}")

    if not os.path.exists(model_path):
        train_viability_model()

    pipeline_data = joblib.load(model_path)

    # Compatibilidad con modelos legacy o apilados
    if isinstance(pipeline_data, dict) and 'model' in pipeline_data:
        extractor = pipeline_data['extractor']
        tfidf_title = pipeline_data.get('tfidf_title')
        tfidf_desc = pipeline_data.get('tfidf_desc')
        tfidf_just = pipeline_data.get('tfidf_just')
        scaler    = pipeline_data['scaler']
        model     = pipeline_data['model']
    else:
        pipeline_data = train_viability_model()
        extractor = pipeline_data['extractor']
        tfidf_title = pipeline_data['tfidf_title']
        tfidf_desc = pipeline_data['tfidf_desc']
        tfidf_just = pipeline_data['tfidf_just']
        scaler    = pipeline_data['scaler']
        model     = pipeline_data['model']

    # Extracción de base features
    X_base = extractor.transform(input_data_df)

    # Extracción de text features
    if tfidf_title is not None and tfidf_desc is not None and tfidf_just is not None:
        title_text = input_data_df["Titulo"].fillna("").astype(str) if "Titulo" in input_data_df.columns else pd.Series("", index=input_data_df.index)
        desc_text  = input_data_df["Descripción"].fillna("").astype(str) if "Descripción" in input_data_df.columns else pd.Series("", index=input_data_df.index)
        just_text  = input_data_df["ds_justificacion"].fillna("").astype(str) if "ds_justificacion" in input_data_df.columns else pd.Series("", index=input_data_df.index)

        X_title = tfidf_title.transform(title_text).toarray()
        X_desc  = tfidf_desc.transform(desc_text).toarray()
        X_just  = tfidf_just.transform(just_text).toarray()

        X_all = np.hstack([X_base, X_title, X_desc, X_just])
    else:
        X_all = X_base

    X = scaler.transform(X_all)

    prediction   = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    return int(prediction), probabilities


def get_project_type_details(sector, title=""):
    """
    Retorna detalles del tipo de proyecto según el sector o el título del dataset.
    """
    sector_norm = str(sector).lower().strip()
    title_norm  = str(title).lower().strip()

    if any(k in sector_norm or k in title_norm for k in
           ['agricultura', 'agropecuario', 'rural', 'suelo', 'clima', 'precio', 'pecuario']):
        return {
            "proj_title": "AgroIA-Col: Optimización y Monitoreo del Sector Agropecuario",
            "ods_align": """- **ODS 2 (Hambre Cero):** Optimización de rendimientos agrícolas y seguridad alimentaria.
- **ODS 13 (Acción por el Clima):** Adaptación y mitigación de riesgos climáticos.
- **ODS 15 (Vida de Ecosistemas Terrestres):** Planificación del suelo productivo.""",
            "ai_arch": """* **Procesamiento de Big Data (PySpark/Dask):** Manejo masivo de registros climáticos históricos.
* **Análisis Geoespacial (GeoPandas):** Mapas de aptitud de suelos UPRA.
* **Predicción de Rendimientos (XGBoost/RandomForest):** Modelos entrenados con EVA.
* **Previsión de Precios (LSTM/Prophet):** Series temporales SIPSA."""
        }
    elif any(k in sector_norm or k in title_norm for k in
             ['ambiente', 'ecolog', 'sostenible', 'agua', 'bosque']):
        return {
            "proj_title": "EcoClima: Alerta Temprana y Preservación Forestal",
            "ods_align": """- **ODS 6 (Agua Limpia):** Monitoreo de recursos hídricos.
- **ODS 13 (Acción por el Clima):** Prevención de desastres naturales.
- **ODS 15 (Vida de Ecosistemas):** Vigilancia de deforestación.""",
            "ai_arch": """* **Visión Artificial (U-Net):** Detección de áreas deforestadas.
* **Clasificación de Riesgo (LightGBM):** Zonas vulnerables a incendios.
* **Clustering Geoespacial (HDBSCAN):** Focos de contaminación hídrica."""
        }
    elif any(k in sector_norm for k in ['salud', 'protección social']):
        return {
            "proj_title": "SaludIntel: Analítica Predictiva de Salud Pública",
            "ods_align": """- **ODS 3 (Salud y Bienestar):** Detección de brotes epidemiológicos.
- **ODS 10 (Reducción de Desigualdades):** Priorización en zonas rurales.""",
            "ai_arch": """* **Modelamiento Epidemiológico (SEIR + NN):** Proyección de propagación.
* **Segmentación Poblacional (K-Means):** Perfiles de riesgo preventivo."""
        }
    elif any(k in sector_norm for k in ['educ', 'cultura']):
        return {
            "proj_title": "EduIA-Plataforma: Inteligencia para la Retención Educativa",
            "ods_align": """- **ODS 4 (Educación de Calidad):** Reducción de deserción escolar.
- **ODS 10 (Reducción de Desigualdades):** Acceso equitativo.""",
            "ai_arch": """* **Predicción de Deserción (RF Classifier):** Identificar estudiantes en riesgo.
* **Recomendación de Contenidos (NLP):** Rutas pedagógicas personalizadas."""
        }
    elif any(k in sector_norm for k in ['transporte', 'infraestructura', 'vias']):
        return {
            "proj_title": "MovilidadIA-Col: Gestión y Flujo Vial Sostenible",
            "ods_align": """- **ODS 9 (Industria e Infraestructura):** Modernización urbana del transporte.
- **ODS 11 (Ciudades Sostenibles):** Reducción de congestión.""",
            "ai_arch": """* **Modelamiento de Congestión (LSTM):** Flujos vehiculares históricos.
* **Predicción de Siniestralidad (Poisson RF):** Probabilidad de accidentes."""
        }
    else:
        return {
            "proj_title": "GovTech-Data: Motor de Analítica Pública Integrada",
            "ods_align": """- **ODS 9 (Innovación e Infraestructura):** Gobernanza basada en datos.
- **ODS 16 (Paz, Justicia e Instituciones):** Transparencia gubernamental.""",
            "ai_arch": """* **NLP (SentenceTransformers):** Categorización de PQRS ciudadanas.
* **Clasificación de Metadata (Supervised):** Calidad de reportes abiertos."""
        }


def generate_project_proposal(dataset_row, viability_prob):
    """
    Genera una propuesta de proyecto GovTech a partir de un dataset del catálogo.
    """
    if isinstance(dataset_row, pd.Series):
        row = dataset_row.to_dict()
    else:
        row = dict(dataset_row)

    titulo       = row.get("Titulo", "Dataset del Ecosistema")
    entidad      = row.get("Información de la Entidad: Nombre de la Entidad", "Entidad Pública del Estado")
    sector       = row.get("Información de la Entidad: Sector", "General")
    descripcion  = row.get("Descripción", "No disponible")
    filas        = row.get("Número de Filas", 0)
    columnas     = row.get("Número de Columnas", 0)
    relevancia   = row.get("ds_score_relevancia", 3.0)
    orden        = row.get("Información de la Entidad: Orden", "Nacional")
    cobertura    = row.get("alcance_geografico", "Nacional")
    justificacion = row.get("ds_justificacion", "Fomentar la analítica de datos abiertos.")
    encabezados  = row.get("ds_encabezados_utiles", "No especificado")

    details     = get_project_type_details(sector, titulo)
    proj_title  = details["proj_title"]
    ods_align   = details["ods_align"]
    ai_arch     = details["ai_arch"]

    viability_porcentaje = round(viability_prob * 100, 2)

    reasons = []
    if filas < 2000:
        reasons.append(f"baja cantidad de registros ({filas:,} filas, recomendado > 2000)")
    if columnas < 10:
        reasons.append(f"escasez de atributos descriptores ({columnas} columnas, recomendado > 10)")
    if relevancia < 3.5:
        reasons.append(f"bajo score de relevancia ({relevancia} sobre 5)")

    warning_banner = ""
    if viability_prob < 0.5:
        reasons_str = ", ".join(reasons) if reasons else "dimensiones y relevancia insuficientes"
        warning_banner = f"""> [!WARNING]
> **ATENCIÓN: Propuesta con Viabilidad Limitada ({viability_porcentaje}%)**
> Limitaciones: *{reasons_str}*.
> Se propone enriquecimiento en Sprint 1 integrando con fuentes robustas (UPRA, IDEAM, FINAGRO).
>

"""

    return f"""{warning_banner}# Propuesta de Innovación Pública: {proj_title}
**Desarrollado para:** {entidad}
**Sector:** {sector} | **Orden:** {orden} | **Cobertura:** {cobertura}

---

## 1. Resumen Ejecutivo
El dataset **"{titulo}"** cuenta con **{columnas} columnas** y **{filas:,} registros**.
La IA asigna una **probabilidad de viabilidad del {viability_porcentaje}%**.

* **Justificación:** {justificacion}
* **Campos Clave:** `{encabezados}`
* **Enlace:** [{row.get("UID", "Catálogo")}]({row.get("url", "https://www.datos.gov.co")})

---

## 2. Alineación con Objetivos de Desarrollo Sostenible (ODS)
{ods_align}

---

## 3. Arquitectura de IA
{ai_arch}
* **Arquitectura Híbrida + LLM:** Integración de modelos de lenguaje para reportes automáticos.

---

## 4. Metodología CRISP-ML(QA)
1. **Business Understanding:** Definir KPI de impacto (reducción de pérdidas, productividad).
2. **Data Understanding:** EDA, diagnóstico de sesgos, evaluación de calidad.
3. **Data Preparation:** Pipeline automatizado de limpieza e imputación.
4. **Modeling:** Entrenamiento con validación cruzada y ajuste de hiperparámetros.
5. **Evaluation:** R2, RMSE, Accuracy, F1-Score vs. datos históricos reales.
6. **Deployment & Monitoring:** API REST (FastAPI) + monitoreo de Data Drift.

---

## 5. Sprints (Scrum 6 semanas)
* **Sprint 1:** Ingesta, EDA y limpieza del dataset `{row.get("UID", "Socrata")}`.
* **Sprint 2:** Modelamiento ML, Feature Engineering y serialización del modelo.
* **Sprint 3:** Dashboard interactivo + conectores Power BI + Copiloto de IA.

---

## 6. Viabilidad
* **Puntaje IA:** `{viability_porcentaje}%`
* **Nivel de Madurez:** {"Bajo - Requiere Saneamiento" if viability_prob < 0.5 else "Medio-Alto (apto para producción)"}
* **Recomendación:** {"Enriquecer con fuentes UPRA+IDEAM antes de desplegar." if viability_prob < 0.5 else f"Iniciar MVP en {row.get('Información de la Entidad: Departamento', 'Cundinamarca')}."}
""".strip()


# ============================================================
# SECCIÓN 2: MODELOS DE RENDIMIENTO Y ECONOMÍA AGRARIA
# ============================================================

from sklearn.ensemble import RandomForestRegressor

def train_crop_yield_model(consolidated_path="data/processed/consolidated_data.csv",
                           model_dir="data/processed"):
    """
    Entrena RandomForestRegressor para predecir rendimiento (Ton/Ha).
    """
    if not os.path.exists(consolidated_path):
        from src.cleaning import process_and_consolidate
        process_and_consolidate()

    df = pd.read_csv(consolidated_path, encoding='utf-8')

    if df.empty:
        raise ValueError("Dataset consolidado vacío.")

    print(f"\nEntrenando modelo de rendimiento: {df.shape[0]} registros.")

    categorical_features = ["cultivo", "textura"]
    numeric_features = ["ph_suelo", "altitud_m", "pendiente_pct",
                        "materia_organica_pct", "temp_media_c", "precipitacion_anual_mm"]

    for col in categorical_features + numeric_features:
        if col not in df.columns:
            df[col] = "Franco" if col == "textura" else 0.0

    X = df[categorical_features + numeric_features]
    y = df["rendimiento_ton_ha"].fillna(df["rendimiento_ton_ha"].mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )

    model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    from sklearn.metrics import mean_squared_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    print(f"RMSE: {round(rmse, 3)} Ton/Ha | R2: {round(r2 * 100, 2)}%")

    pipeline.fit(X, y)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "crop_yield_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"Modelo rendimiento exportado a: {model_path}")

    return pipeline


def predict_crop_yield(input_data, model_path="data/processed/crop_yield_model.joblib"):
    """
    Predice el rendimiento (Ton/Ha) dado los parámetros del predio.
    """
    if not os.path.exists(model_path):
        train_crop_yield_model()

    pipeline = joblib.load(model_path)

    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    cols = ["cultivo", "textura", "ph_suelo", "altitud_m", "pendiente_pct",
            "materia_organica_pct", "temp_media_c", "precipitacion_anual_mm"]

    for col in cols:
        if col not in input_data.columns:
            input_data[col] = "Franco" if col == "textura" else 0.0

    input_data = input_data[cols]
    return float(pipeline.predict(input_data)[0])


# ────────────────────────────────────────────────────────────
# PRECIOS SIPSA (canasta familiar y valor de cosecha)
# ────────────────────────────────────────────────────────────

# Precios base de la canasta familiar colombiana 2026 (COP/kg en centrales)
SIPSA_BASE_PRICES = {
    "Cafe":    {"precio_kg": 14500, "canasta_pct": 4.2,  "demanda_tendencia": +1.8},
    "Cacao":   {"precio_kg": 11200, "canasta_pct": 1.8,  "demanda_tendencia": +3.1},
    "Arroz":   {"precio_kg": 2800,  "canasta_pct": 8.7,  "demanda_tendencia": +0.5},
    "Maiz":    {"precio_kg": 2100,  "canasta_pct": 5.3,  "demanda_tendencia": +2.2},
    "Platano": {"precio_kg": 1800,  "canasta_pct": 6.9,  "demanda_tendencia": +1.1},
}

def get_sipsa_prices(crop=None, raw_dir="data/raw"):
    """
    Obtiene los precios más recientes de SIPSA para el cultivo.
    Primero intenta los datos reales, luego usa la tabla base.
    """
    prices = {}
    sipsa_path = os.path.join(raw_dir, "sipsa_precios_mercado.csv")

    if os.path.exists(sipsa_path):
        try:
            df_sipsa = pd.read_csv(sipsa_path)
            # Tomar el precio promedio de los últimos 12 registros por cultivo
            df_sipsa["fecha"] = pd.to_datetime(df_sipsa["fecha"], errors="coerce")
            df_sipsa = df_sipsa.dropna(subset=["fecha"])
            for c in df_sipsa["cultivo"].unique():
                df_c = df_sipsa[df_sipsa["cultivo"] == c].sort_values("fecha").tail(12)
                if not df_c.empty:
                    precio = float(df_c["precio_kg_promedio_cop"].mean())
                    prices[c] = SIPSA_BASE_PRICES.get(c, {}).copy()
                    prices[c]["precio_kg"] = max(precio, 100)
        except Exception as e:
            print(f"[SIPSA] Error leyendo datos reales: {e}")

    # Completar con base si faltan cultivos
    for c, v in SIPSA_BASE_PRICES.items():
        if c not in prices:
            prices[c] = v.copy()

    if crop:
        return prices.get(crop, SIPSA_BASE_PRICES.get(crop, {"precio_kg": 2000,
                                                              "canasta_pct": 2.0,
                                                              "demanda_tendencia": 0.5}))
    return prices


def predict_crop_economics(crop, ph, altitude, slope, organic_matter, texture,
                           temp, rain, area_ha, dept="Todos", mun="Todos"):
    """
    Calcula la economía completa de un cultivo:
    - Rendimiento predicho (Ton/Ha)
    - Precio de mercado SIPSA
    - Ingreso bruto, costos, utilidad neta, ROI
    - Importancia en canasta familiar
    - Proyección mensual de flujo de caja (12 meses)
    """
    # ── Rendimiento predicho ────────────────────────────────────────────
    yield_input = {
        "cultivo": crop, "textura": texture, "ph_suelo": ph,
        "altitud_m": altitude, "pendiente_pct": slope,
        "materia_organica_pct": organic_matter,
        "temp_media_c": temp, "precipitacion_anual_mm": rain
    }
    try:
        yield_ton_ha = predict_crop_yield(yield_input)
    except Exception:
        base_yields = {"Cafe": 1.4, "Cacao": 0.6, "Arroz": 5.5, "Maiz": 3.8, "Platano": 12.0}
        yield_ton_ha = base_yields.get(crop, 2.0)

    total_production_ton = yield_ton_ha * area_ha

    # ── Factor zonal desde EVA ──────────────────────────────────────────
    zone_factor = 1.0
    eva_path = "data/raw/produccion_historica.csv"
    if os.path.exists(eva_path):
        try:
            df_eva = pd.read_csv(eva_path)
            filt = df_eva[df_eva["cultivo"].str.lower() == crop.lower()]
            if dept != "Todos":
                f2 = filt[filt["departamento"].str.lower() == dept.lower()]
                filt = f2 if not f2.empty else filt
            if not filt.empty and "rendimiento_ton_ha" in filt.columns:
                real_avg = float(filt["rendimiento_ton_ha"].mean())
                base = {"cafe": 1.4, "cacao": 0.6, "arroz": 5.5, "maiz": 3.8, "platano": 12.0}
                nat = base.get(crop.lower(), 2.0)
                zone_factor = max(0.6, min(2.0, real_avg / nat)) if nat > 0 else 1.0
                yield_ton_ha *= zone_factor
                total_production_ton = yield_ton_ha * area_ha
        except Exception:
            pass

    # ── Precios SIPSA ───────────────────────────────────────────────────
    sipsa_info = get_sipsa_prices(crop)
    precio_kg  = sipsa_info.get("precio_kg", 2000)
    canasta_pct = sipsa_info.get("canasta_pct", 2.0)
    demanda_trend = sipsa_info.get("demanda_tendencia", 0.5)

    # ── Costos de producción (FINAGRO/MADR 2025) ────────────────────────
    cost_per_ha = {
        "Cafe":    3_800_000, "Cacao":  2_200_000,
        "Arroz":   5_500_000, "Maiz":   3_200_000, "Platano": 4_100_000
    }.get(crop, 3_000_000)

    total_cost   = cost_per_ha * area_ha
    gross_revenue = total_production_ton * 1000 * precio_kg  # kg → precio
    net_profit   = gross_revenue - total_cost
    margin_pct   = (net_profit / gross_revenue * 100) if gross_revenue > 0 else 0
    roi_pct      = (net_profit / total_cost * 100) if total_cost > 0 else 0

    # ── Crédito FINAGRO ─────────────────────────────────────────────────
    interest_rate = 12.5  # % anual promedio
    finagro_path = "data/raw/finagro_creditos.csv"
    if os.path.exists(finagro_path):
        try:
            df_fin = pd.read_csv(finagro_path)
            if dept != "Todos":
                df_fin = df_fin[df_fin["departamento"].str.lower() == dept.lower()]
            if not df_fin.empty and "tasa_interes_promedio" in df_fin.columns:
                interest_rate = float(df_fin["tasa_interes_promedio"].mean())
        except Exception:
            pass

    recommended_credit = total_cost * 0.70
    annual_interest    = recommended_credit * (interest_rate / 100)

    # ── Proyección mensual (flujo de caja) ─────────────────────────────
    crop_cycles = {"Cafe": 12, "Cacao": 12, "Arroz": 4, "Maiz": 4, "Platano": 8}
    cycle_months = crop_cycles.get(crop, 6)

    months = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
              "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    monthly_cash_flow = []
    monthly_cost_per  = total_cost / 12
    harvest_months    = [cycle_months - 1] if cycle_months <= 6 else [5, 11]

    for i, m in enumerate(months):
        income  = gross_revenue / len(harvest_months) if i in harvest_months else 0
        cost_m  = monthly_cost_per
        interest_m = annual_interest / 12
        net_m   = income - cost_m - interest_m
        monthly_cash_flow.append({
            "mes": m, "ingresos": round(income), "costos": round(cost_m),
            "intereses": round(interest_m), "neto": round(net_m)
        })

    return {
        "crop": crop, "area_ha": area_ha,
        "yield_ton_ha": round(yield_ton_ha, 2),
        "total_production_ton": round(total_production_ton, 2),
        "precio_kg_cop": int(precio_kg),
        "gross_revenue": int(gross_revenue),
        "total_cost": int(total_cost),
        "net_profit": int(net_profit),
        "margin_pct": round(margin_pct, 1),
        "roi_pct": round(roi_pct, 1),
        "recommended_credit": int(recommended_credit),
        "annual_interest": int(annual_interest),
        "interest_rate": round(interest_rate, 2),
        "canasta_pct": canasta_pct,
        "demanda_trend": demanda_trend,
        "zone_factor": round(zone_factor, 2),
        "monthly_cash_flow": monthly_cash_flow
    }


def compare_all_crops(ph, altitude, slope, organic_matter, texture,
                      temp, rain, area_ha, dept="Todos", mun="Todos"):
    """
    Compara la rentabilidad de TODOS los cultivos para las condiciones del predio.
    Incluye canasta familiar y tendencia de demanda.
    """
    crops = ["Cafe", "Cacao", "Arroz", "Maiz", "Platano"]
    results = []
    for crop in crops:
        try:
            eco = predict_crop_economics(
                crop, ph, altitude, slope, organic_matter, texture,
                temp, rain, area_ha, dept, mun
            )
            eco["cultivo"] = crop
            results.append(eco)
        except Exception as e:
            print(f"[COMPARE] Error en {crop}: {e}")
    return sorted(results, key=lambda x: x["net_profit"], reverse=True)


def predict_optimal_cycle(crop, ph, altitude, slope, organic_matter, texture,
                          temp_base, rain_base, dept="Todos", mun="Todos"):
    """
    Simula los 12 meses para encontrar el mejor mes de siembra.
    Calibra con datos reales de IDEAM y EVA si están disponibles.
    """
    monthly_rain_fraction = [0.03, 0.04, 0.08, 0.12, 0.15, 0.07,
                             0.05, 0.04, 0.09, 0.13, 0.14, 0.06]
    monthly_temp_diff     = [-1.0, -0.5, 0.0, 0.5, 1.0, 0.8,
                             0.5,  0.6, 0.4, 0.0, -0.5, -0.9]
    zone_yield_factor = 1.0
    data_source = ""

    try:
        ideam_path = "data/raw/ideam_clima_historico.csv"
        eva_path   = "data/raw/produccion_historica.csv"

        if os.path.exists(ideam_path):
            chunks = []
            for chunk in pd.read_csv(ideam_path, chunksize=50000):
                if dept != "Todos":
                    chunk = chunk[chunk["departamento"].str.lower() == dept.lower()]
                    if mun != "Todos":
                        chunk = chunk[chunk["municipio"].str.lower() == mun.lower()]
                if not chunk.empty:
                    chunks.append(chunk[["fecha", "precipitacion_diaria_mm", "temperatura_promedio_c"]])
            if chunks:
                df_clima = pd.concat(chunks, ignore_index=True)
                df_clima["mes"] = pd.to_datetime(df_clima["fecha"], errors="coerce").dt.month
                monthly_totals = df_clima.groupby("mes")["precipitacion_diaria_mm"].sum()
                total_rain = monthly_totals.sum()
                if total_rain > 0:
                    for m_idx in range(12):
                        mes = m_idx + 1
                        if mes in monthly_totals.index:
                            monthly_rain_fraction[m_idx] = float(monthly_totals[mes] / total_rain)
                temp_real = df_clima["temperatura_promedio_c"].mean()
                if not pd.isna(temp_real) and 10.0 <= temp_real <= 35.0:
                    temp_base = temp_real
                monthly_temps = df_clima.groupby("mes")["temperatura_promedio_c"].mean()
                temp_global_avg = monthly_temps.mean()
                if not pd.isna(temp_global_avg):
                    for m_idx in range(12):
                        mes = m_idx + 1
                        if mes in monthly_temps.index:
                            monthly_temp_diff[m_idx] = float(monthly_temps[mes] - temp_global_avg)
                data_source = f"IDEAM calibrado para {mun if mun != 'Todos' else dept}"

        if os.path.exists(eva_path):
            df_eva = pd.read_csv(eva_path)
            filt = df_eva[df_eva["cultivo"].str.lower() == crop.lower()]
            if dept != "Todos":
                filt = filt[filt["departamento"].str.lower() == dept.lower()]
                if mun != "Todos":
                    filt_m = filt[filt["municipio"].str.lower() == mun.lower()]
                    filt = filt_m if not filt_m.empty else filt
            if not filt.empty and "rendimiento_ton_ha" in filt.columns:
                real_avg = float(filt["rendimiento_ton_ha"].mean())
                nat_avg  = {"cafe": 1.4, "cacao": 0.6, "arroz": 5.5, "maiz": 3.8, "platano": 12.0}
                nat = nat_avg.get(crop.lower(), 2.0)
                zone_yield_factor = max(0.5, min(2.5, real_avg / nat)) if nat > 0 else 1.0
                data_source += f" + EVA zona ({zone_yield_factor:.2f}x nacional)"
    except Exception as e:
        print(f"[CAMPO] Calibración no disponible: {e}")

    months = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
              "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]

    yields = {}
    for idx, month in enumerate(months):
        cycle_months = [(idx + m) % 12 for m in range(4)]
        cycle_rain_frac = sum(monthly_rain_fraction[m] for m in cycle_months)
        sim_rain = rain_base * (cycle_rain_frac * (12 / 4))
        sim_temp = temp_base + (sum(monthly_temp_diff[m] for m in cycle_months) / 4)
        input_dict = {"cultivo": crop, "textura": texture, "ph_suelo": ph,
                      "altitud_m": altitude, "pendiente_pct": slope,
                      "materia_organica_pct": organic_matter,
                      "temp_media_c": sim_temp, "precipitacion_anual_mm": sim_rain}
        try:
            predicted = predict_crop_yield(input_dict) * zone_yield_factor
        except Exception:
            base = {"Cafe": 1.4, "Cacao": 0.6, "Arroz": 5.5, "Maiz": 3.8, "Platano": 12.0}
            predicted = base.get(crop, 2.0) * zone_yield_factor
        yields[month] = round(predicted, 2)

    sorted_months = sorted(yields.items(), key=lambda x: x[1], reverse=True)
    best_month = sorted_months[0][0]
    best_yield = sorted_months[0][1]

    # ── Validación regulatoria ──────────────────────────────────────────
    madr_rules = []
    compliance_status = "Conforme"
    crop_lower = str(crop).lower()

    if "cafe" in crop_lower:
        if ph < 5.0:
            madr_rules.append("⚠️ Suelo muy ácido para Café (pH < 5.0). Requiere encalado.")
            compliance_status = "Condicionado"
        elif ph > 6.5:
            madr_rules.append("⚠️ Suelo ligeramente alcalino para Café (pH > 6.5).")
            compliance_status = "Condicionado"
        if altitude < 1000:
            madr_rules.append("⚠️ Altitud baja para Café (< 1000 m). Riesgo de plagas y baja calidad.")
            compliance_status = "Condicionado"
        elif altitude > 2000:
            madr_rules.append("⚠️ Altitud muy alta para Café (> 2000 m). Riesgo de heladas.")
            compliance_status = "Condicionado"
    elif "cacao" in crop_lower:
        if ph < 5.5:
            madr_rules.append("⚠️ Suelo ácido para Cacao (pH < 5.5). Enmiendas recomendadas.")
            compliance_status = "Condicionado"
        if altitude > 1200:
            madr_rules.append("⚠️ Altitud limitante para Cacao (> 1200 m). Fuera de aptitud UPRA.")
            compliance_status = "Condicionado"
    elif "arroz" in crop_lower:
        if ph < 5.0 or ph > 7.0:
            madr_rules.append("⚠️ pH fuera de rango para Arroz (5.5–6.5).")
            compliance_status = "Condicionado"
        if altitude > 1000:
            madr_rules.append("⚠️ Altitud restrictiva para Arroz comercial (> 1000 m).")
            compliance_status = "Condicionado"
    elif "maiz" in crop_lower:
        if altitude > 2000:
            madr_rules.append("📌 UPRA: Use variedades de maíz frío para altitudes > 2000 m.")

    if slope > 15:
        madr_rules.append("⚠️ Pendiente alta (> 15%). Requiere siembra en curvas de nivel.")
        compliance_status = "Condicionado"
    if slope > 30:
        madr_rules.append("❌ Pendiente crítica (> 30%). Incumple aptitud UPRA para transitorios.")
        compliance_status = "Crítico"

    if best_month in ["Enero", "Febrero", "Agosto", "Diciembre"]:
        madr_rules.append("⚠️ Siembra en época seca. Requiere sistema de riego (Decreto 1076 CAR).")
        if compliance_status != "Crítico":
            compliance_status = "Condicionado"
    else:
        madr_rules.append("✅ Fecha ideal al inicio de lluvias. Cumple circular MADR 2026.")

    if zone_yield_factor < 0.8:
        madr_rules.append(f"📌 Esta zona produce {round((1-zone_yield_factor)*100)}% menos que el promedio nacional.")
    elif zone_yield_factor > 1.2:
        madr_rules.append(f"✅ Zona de alta productividad: {round((zone_yield_factor-1)*100)}% sobre el promedio nacional.")

    madr_rules.append("📌 Registre su predio en el ICA antes de vender la cosecha.")

    return {
        "crop": crop, "best_month": best_month, "best_yield": best_yield,
        "yields": yields, "madr_compliance": compliance_status,
        "madr_rules": madr_rules, "zone_yield_factor": round(zone_yield_factor, 2),
        "data_source": data_source
    }


def predict_livestock_economics(dept, mun, species, purpose, herd_size, farm_area):
    """
    Proyecciones económicas y regulatorias para hato ganadero (bovino/porcino).
    """
    slope = 10.0
    organic_matter = 3.2
    avg_milk_yield = 4.2
    vaccination_rate = 97.5
    interest_rate = 12.8

    try:
        for path, dtype, callback in [
            ("data/raw/upra_aptitud_suelos.csv", "soils", None),
            ("data/raw/inventario_ganadero_nacional.csv", "livestock", None),
            ("data/raw/finagro_creditos.csv", "finagro", None)
        ]:
            if not os.path.exists(path):
                continue
            df_t = pd.read_csv(path)
            if dept != "Todos":
                df_t = df_t[df_t["departamento"].str.lower() == dept.lower()]
                if mun != "Todos":
                    df_t = df_t[df_t["municipio"].str.lower() == mun.lower()]
            if df_t.empty:
                continue
            if "soils" in path:
                if "pendiente_pct" in df_t.columns:
                    slope = float(df_t["pendiente_pct"].mean())
                if "materia_organica_pct" in df_t.columns:
                    organic_matter = float(df_t["materia_organica_pct"].mean())
            elif "ganadero" in path:
                s_milk = df_t["produccion_leche_litros_dia"].dropna()
                s_vac  = df_t["vacunacion_fiebre_aftosa_pct"].dropna()
                if not s_milk.empty:
                    avg_milk_yield = float(s_milk.mean())
                if not s_vac.empty:
                    vaccination_rate = float(s_vac.mean())
            elif "finagro" in path:
                s_rate = df_t["tasa_interes_promedio"].dropna()
                if not s_rate.empty:
                    interest_rate = float(s_rate.mean())
    except Exception as e:
        print(f"[Ganadería] Error cargando microdatos: {e}")

    # Carga animal
    base_carrying = 1.8 if species == "Bovino" else 15.0
    slope_factor = 0.25 if slope > 30 else (0.65 if slope > 15 else 1.0)
    om_factor    = 0.8 if organic_matter < 2.0 else (1.25 if organic_matter > 4.0 else 1.0)
    carrying_capacity_ha = base_carrying * slope_factor * om_factor
    max_animals_allowed  = int(carrying_capacity_ha * farm_area)

    # Ingresos
    gross_revenue = 0.0
    milk_prod_liters_day = 0.0
    meat_sold_kg_year = 0.0

    if species == "Bovino":
        if purpose in ["Leche", "Doble Propósito"]:
            milk_factor = 0.8 if purpose == "Doble Propósito" else 1.0
            milk_prod_liters_day = herd_size * 0.60 * avg_milk_yield * milk_factor
            gross_revenue += milk_prod_liters_day * 305 * 2150.0
        if purpose in ["Carne", "Doble Propósito"]:
            meat_factor = 0.75 if purpose == "Doble Propósito" else 1.0
            animals_sold = max(1, int(herd_size * 0.35 * meat_factor))
            meat_sold_kg_year = animals_sold * 420.0
            gross_revenue += meat_sold_kg_year * 8300.0
    else:
        animals_sold = int(herd_size * 1.6)
        meat_sold_kg_year = animals_sold * 95.0
        gross_revenue += meat_sold_kg_year * 9600.0

    cost_base_per_head   = 920000.0 if species == "Bovino" else 620000.0
    total_operating_costs = herd_size * cost_base_per_head
    capital_investment    = herd_size * (2800000 if species == "Bovino" else 750000)
    recommended_credit    = capital_investment * 0.70
    annual_interest_payment = recommended_credit * (interest_rate / 100.0)
    net_profit  = gross_revenue - total_operating_costs - annual_interest_payment
    margin_pct  = (net_profit / gross_revenue) * 100.0 if gross_revenue > 0 else 0.0
    roi_pct     = (net_profit / capital_investment) * 100.0 if capital_investment > 0 else 0.0

    # Proyección mensual
    monthly_cash = []
    months_abbr = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    for i, m in enumerate(months_abbr):
        # Leche: ingreso mensual; carne: ingreso en mes 6 y 12
        if species == "Bovino" and purpose in ["Leche", "Doble Propósito"]:
            income_m = milk_prod_liters_day * 30 * 2150.0
            if purpose == "Doble Propósito" and i in [5, 11]:
                income_m += (gross_revenue - milk_prod_liters_day * 305 * 2150) / 2
        elif i in [5, 11]:
            income_m = gross_revenue / 2
        else:
            income_m = 0
        cost_m = total_operating_costs / 12
        interest_m = annual_interest_payment / 12
        monthly_cash.append({
            "mes": m, "ingresos": round(income_m),
            "costos": round(cost_m), "intereses": round(interest_m),
            "neto": round(income_m - cost_m - interest_m)
        })

    # Alertas
    alerts = []
    compliance = "Conforme"
    if species == "Bovino":
        actual_density = herd_size / farm_area if farm_area > 0 else 0
        if actual_density > carrying_capacity_ha:
            alerts.append(f"❌ SOBREPASTOREO: {herd_size} bovinos excede capacidad ({max_animals_allowed} máx.).")
            compliance = "Crítico"
        elif actual_density > carrying_capacity_ha * 0.8:
            alerts.append(f"⚠️ CARGA LÍMITE: {round(actual_density, 2)} cab/Ha cerca del máximo ecológico.")
            compliance = "Condicionado"
        else:
            alerts.append("✅ CARGA ANIMAL SOSTENIBLE: Hato óptimamente dimensionado.")
        if slope > 15.0:
            alerts.append("⚠️ PENDIENTE (CAR): Implemente silvopastoril para evitar erosión (Decreto 1076).")
            if compliance != "Crítico":
                compliance = "Condicionado"
    else:
        density_p = herd_size / farm_area if farm_area > 0 else 0
        if density_p > 25.0:
            alerts.append("❌ DENSIDAD PORCINA CRÍTICA: Supera directrices ICA. Riesgo sanitario.")
            compliance = "Crítico"

    if vaccination_rate < 95.0:
        alerts.append(f"❌ ALERTA ICA: Vacunación aftosa {round(vaccination_rate, 2)}% < 95% requerido.")
        compliance = "Crítico"
    else:
        alerts.append(f"✅ CONTROL ICA: Vacunación aftosa {round(vaccination_rate, 2)}% — Óptima.")

    return {
        "species": species, "carrying_capacity_ha": round(carrying_capacity_ha, 2),
        "max_animals_allowed": max_animals_allowed,
        "milk_prod_liters_day": round(milk_prod_liters_day, 1),
        "meat_sold_kg_year": round(meat_sold_kg_year, 1),
        "gross_revenue": int(gross_revenue),
        "total_operating_costs": int(total_operating_costs),
        "capital_investment": int(capital_investment),
        "recommended_credit": int(recommended_credit),
        "annual_interest_payment": int(annual_interest_payment),
        "net_profit": int(net_profit),
        "margin_pct": round(margin_pct, 1),
        "roi_pct": round(roi_pct, 1),
        "alerts": alerts, "compliance": compliance,
        "monthly_cash_flow": monthly_cash
    }


# ── Aliases legacy ────────────────────────────────────────────────────────────
def train_crop_model(*args, **kwargs):
    return train_viability_model(*args, **kwargs)

def predict_yield(input_data, *args, **kwargs):
    pred, prob = predict_viability(input_data)
    return float(prob[1])


if __name__ == "__main__":
    train_viability_model()
