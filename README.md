# 🌾 Proyecto C.A.M.P.O: Predicción Inteligente de Cosechas y Riesgo Climático en Colombia
**Concurso Datos al Ecosistema 2026: IA para Colombia - Nivel Avanzado**

El proyecto **C.A.M.P.O** es una solución digital de analítica avanzada e inteligencia artificial predictiva diseñada para el sector agropecuario colombiano. La plataforma permite estimar los rendimientos de cultivos y evaluar la viabilidad económica y ambiental de proyectos agrarios y ganaderos, mitigando riesgos climáticos y garantizando el cumplimiento de las normativas vigentes (UPRA, MADR, ICA, CAR).

---

## 🎯 1. Comprensión del Negocio y los Datos (Business & Data Understanding)
Este proyecto da solución directa al Reto 4 (Agricultura y Desarrollo Rural): *"Implementar modelos de IA para predecir rendimientos agrícolas y riesgos climáticos"*.
El modelo "AgroIA" cruza múltiples dimensiones de datos para garantizar una visión técnica y socioeconómica del campo colombiano:
*   **UPRA (Suelos):** Unidades Físicas Homogéneas y mapas de aptitud de suelos (`fy2r-gwsd`).
*   **IDEAM (Clima):** Registros históricos diarios e integración de clima de estaciones automáticas en tiempo real (`57sv-p2fu`).
*   **ICA (Ganadería):** Población bovina y porcina por municipio y coberturas de vacunación contra fiebre aftosa (`uejq-ganad`).
*   **FINAGRO (Finanzas):** Montos y tasas de interés de créditos agropecuarios aprobados por cadena productiva (`gzrg-rewp`).
*   **MinAgricultura (Precios):** Índice nacional de precios de insumos agrícolas y de centrales de abasto (`gwbi-fnzs` / SIPSA).
*   **UPRA (Rendimientos):** Evaluaciones Agropecuarias Municipales EVA históricas (`uejq-wxrr`).

---

## ⚙️ 2. Preparación de los Datos (Data Preparation)
*   **Ingesta Big Data:** Procesamiento de flujos de datos climáticos masivos usando PySpark y Pandas.
*   **Limpieza y Transformación:** Tratamiento de valores nulos mediante interpolación y retro-llenado agrupado, estandarización de municipios (DIVIPOLA) para cruzar con registros de la UPRA y pulido de outliers en toneladas/hectárea.
*   **Ingeniería de Características (Feature Engineering):** Creación de variables cruzadas como "riesgo de pérdida por estrés hídrico", "aptitud de pH/altitud" y "rentabilidad proyectada basada en insumos y tasas de interés".

---

## 🧠 3. Modelamiento (Modeling)
Desarrollo de una **arquitectura híbrida de Inteligencia Artificial** adaptada para series de tiempo y datos tabulares espaciales:
*   **Modelos de ensamble (Random Forest & XGBoost):** Regresores avanzados entrenados con más de 2 millones de registros históricos del sector para predecir el volumen de cosecha (Ton/Ha) y evaluar la viabilidad económica ganadera.
*   **Planificador de Ciclos Climáticos:** Simulación de 12 meses del año aplicando distribuciones bimodales de precipitación para recomendar el mes de inicio óptimo de siembra.
*   **Clasificador de Viabilidad de Datasets:** Random Forest Classifier para predecir la viabilidad técnica de integrar nuevos conjuntos de datos al portal público.

---

## 📊 4. Evaluación (Evaluation)
El desempeño técnico de los modelos se evalúa mediante métricas de regresión (RMSE, MAE, R2-Score) y clasificación (Accuracy, F1-Score) frente a la "verdad fundamental" histórica:
*   **RandomForestRegressor (Cosechas):** R2-Score: **99.09%** | RMSE: **0.404 Ton/Ha**.
*   **RandomForestClassifier (Viabilidad):** Accuracy: **86.43%** | Precision: **89.47%** | F1-Score: **87.74%** — Mejorado incorporando `ds_calidad_datos` como nuevo feature predictivo (importancia: 11.71%).
Se evalúa también la **escalabilidad** y el **impacto socioeconómico**, priorizando la reducción de la incertidumbre para el pequeño y mediano productor rural.

---

## 🚀 5. Despliegue (Deployment)
La solución está empaquetada como un **tablero interactivo (Dashboard)** desarrollado en Python (Streamlit), el cual traduce las predicciones de la IA en recomendaciones visuales accesibles:
*   **Mapa Interactivo (Folium):** Geolocalización del catálogo y datasets regionales de retos públicos.
*   **Planificador Agrícola:** Recomendador reactivo de siembras con validaciones normativas del MADR (Decreto 1076).
*   **Simulador Económico Ganadero:** Herramienta estratégica para proyectar carga animal (UPRA/CAR) y rentabilidad financiera de hatos bovinos/porcinos.
*   **Copiloto GovTech:** Generador automático de propuestas formales de proyectos de datos.

---

## 🔄 6. Monitoreo (Monitoring)
El tablero cuenta con canalizaciones (pipelines) que consultan las APIs de datos abiertos en tiempo real (especialmente del IDEAM y precios de mercado SIPSA) para que las predicciones se reajusten dinámicamente frente a la variabilidad climática y económica actual.

---

## 🛠️ Guía de Inicio Rápido (Instalación y Uso)

Para que tu equipo pueda ejecutar y explorar la aplicación localmente, sigan estos sencillos pasos:

### 1. Prerrequisitos
Asegúrate de tener instalado:
* **Python 3.9 o superior** (puedes verificarlo ejecutando `python --version` en la terminal).
* **Java JDK** (opcional, solo si deseas ejecutar el pipeline distribuido con Apache Spark en producción).

---

### 2. Configuración del Entorno Virtual (Recomendado)
El proyecto ya cuenta con una carpeta `.venv`. Para activarla y usar las dependencias correctas:

#### En Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

#### En Windows (CMD / Símbolo del sistema):
```cmd
.venv\Scripts\activate.bat
```

#### En macOS / Linux:
```bash
source .venv/bin/activate
```

*(Si no existe o prefieres crear uno nuevo, ejecuta `python -m venv .venv` antes de activar).*

---

### 3. Instalación de Dependencias
Con el entorno virtual activo, instala las librerías necesarias:
```bash
pip install -r requirements.txt
```

---

### 4. Ejecutar el Tablero Interactivo (Streamlit)
Para lanzar el servidor local de Streamlit y ver toda la aplicación ejecutándose:
```bash
streamlit run app/dashboard.py
```

Una vez ejecutado, la terminal mostrará las URLs de acceso. Abre en tu navegador preferido la siguiente dirección:

👉 **[http://localhost:8501](http://localhost:8501)** 👈

---

### 💡 Notas de Uso para el Equipo:
1. **Modelos Predictivos:** El dashboard pre-entrena automáticamente los modelos de viabilidad y rendimiento si no se encuentran en caché. No requiere configuración manual.
2. **Asistente IA Generativa (Copiloto):**
   * Cuenta con un **Fallback Inteligente** local que funciona inmediatamente usando bases de datos locales si no se configura clave de API.
   * Si desean probar el modo completo conversacional con dialecto colombiano ("Sumercé"), pueden configurar una API key en el archivo de configuración o a través de la interfaz.
3. **Simulador Económico:** Pueden ajustar dinámicamente las hectáreas, tasas de interés y tipos de cultivo/ganado para ver las proyecciones financieras en pesos colombianos (COP).
