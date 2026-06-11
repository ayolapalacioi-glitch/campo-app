import os
import unittest
import pandas as pd
import joblib
from src.models import predict_viability, generate_project_proposal

class TestAgroPredictPipeline(unittest.TestCase):
    
    def setUp(self):
        self.workspace_dir = "."
        self.processed_dir = "data/processed"
        self.model_path = os.path.join(self.processed_dir, "dataset_viability_model.joblib")
        self.catalog_path = os.path.join(self.processed_dir, "cleaned_datasets_catalog.csv")
        
    def test_challenge_catalog_raw_exists(self):
        """Verifica que el catálogo del reto original enviado esté en el directorio raíz."""
        path = os.path.join(self.workspace_dir, "reto_04_agricultura_20260430_225257.csv")
        self.assertTrue(os.path.exists(path), "El catálogo crudo reto_04_agricultura_20260430_225257.csv no está en la raíz.")
        
    def test_processed_catalog_exists_and_geocoded(self):
        """Verifica que el catálogo limpio y georreferenciado contenga coordenadas y esté completo."""
        self.assertTrue(os.path.exists(self.catalog_path), "Falta el catálogo limpio cleaned_datasets_catalog.csv")
        df = pd.read_csv(self.catalog_path)
        self.assertGreater(len(df), 0, "El catálogo procesado está vacío.")
        self.assertIn("latitud", df.columns, "Falta la columna latitud")
        self.assertIn("longitud", df.columns, "Falta la columna longitud")
        # Asegurar que las coordenadas no tengan nulos
        self.assertEqual(df["latitud"].isnull().sum(), 0, "Hay coordenadas nulas en latitud")
        self.assertEqual(df["longitud"].isnull().sum(), 0, "Hay coordenadas nulas en longitud")
        
    def test_model_exists_and_loads(self):
        """Verifica que el clasificador de viabilidad se cargue correctamente."""
        self.assertTrue(os.path.exists(self.model_path), "Falta el archivo del modelo dataset_viability_model.joblib")
        pipeline = joblib.load(self.model_path)
        self.assertIsNotNone(pipeline)
        
    def test_viability_prediction_function(self):
        """Verifica que la inferencia de clasificación de viabilidad retorne valores válidos (0 o 1) y probabilidades."""
        sample_input = {
            "alcance_geografico": "nacional",
            "Información de la Entidad: Sector": "Tecnologías de la Información y Comunicaciones",
            "Información de la Entidad: Orden": "Nacional",
            "Número de Filas": 5000,
            "Número de Columnas": 15,
            "ds_score_relevancia": 4.5
        }
        pred, probs = predict_viability(sample_input, self.model_path)
        self.assertIn(pred, [0, 1], "La predicción debe ser 0 o 1.")
        self.assertEqual(len(probs), 2, "Debe retornar las probabilidades para ambas clases (0 y 1).")
        self.assertAlmostEqual(sum(probs), 1.0, places=4, msg="Las probabilidades deben sumar 1.0")

    def test_proposal_generation(self):
        """Verifica que la función generate_project_proposal cree una propuesta estructurada en formato markdown."""
        sample_row = {
            "Titulo": "Reporte de Cultivos de Café",
            "Información de la Entidad: Nombre de la Entidad": "Alcaldía de Pitalito",
            "Información de la Entidad: Sector": "Agricultura y Desarrollo Rural",
            "Descripción": "Información sobre hectáreas sembradas y producción de café.",
            "Número de Filas": 1500,
            "Número de Columnas": 8,
            "ds_score_relevancia": 4.2,
            "Información de la Entidad: Orden": "Territorial",
            "alcance_geografico": "Municipal",
            "ds_justificacion": "Datos oficiales de café regional.",
            "ds_encabezados_utiles": "hectareas, produccion, municipio"
        }
        
        proposal = generate_project_proposal(sample_row, 0.885)
        self.assertIsInstance(proposal, str)
        self.assertIn("# Propuesta de Innovación Pública:", proposal)
        self.assertIn("Resumen Ejecutivo", proposal)
        self.assertIn("Alineación con Objetivos de Desarrollo Sostenible (ODS)", proposal)
        self.assertIn("CRISP-ML", proposal)
        self.assertIn("Scrum", proposal)

    def test_crop_yield_and_cycle_planner(self):
        """Verifica que el modelo predictivo de rendimiento y el planificador de ciclo óptimo con altitud funcionen."""
        from src.models import predict_crop_yield, predict_optimal_cycle
        
        # Probar predict_crop_yield con diccionario de entrada
        sample_input = {
            "cultivo": "Cafe",
            "textura": "Franco",
            "ph_suelo": 5.8,
            "altitud_m": 1200.0,
            "pendiente_pct": 12.0,
            "materia_organica_pct": 3.5,
            "temp_media_c": 21.0,
            "precipitacion_anual_mm": 1800.0
        }
        yield_pred = predict_crop_yield(sample_input)
        self.assertIsInstance(yield_pred, float)
        self.assertGreater(yield_pred, 0.0, "El rendimiento estimado debe ser un número real positivo.")
        
        # Probar predict_optimal_cycle
        res = predict_optimal_cycle(
            crop="Cafe",
            ph=5.8,
            altitude=1200.0,
            slope=12.0,
            organic_matter=3.5,
            texture="Franco",
            temp_base=21.0,
            rain_base=1800.0
        )
        self.assertIsInstance(res, dict)
        self.assertEqual(res["crop"], "Cafe")
        self.assertIn("best_month", res)
        self.assertIn("best_yield", res)
        self.assertIn("yields", res)
        self.assertIn("madr_compliance", res)
        self.assertIn("madr_rules", res)
        self.assertEqual(len(res["yields"]), 12, "Debe simular exactamente 12 meses de siembra.")
        
        # Verificar formato XAI
        self.assertIsInstance(res["xai_explanation"], dict)
        self.assertIn("resumen", res["xai_explanation"])
        self.assertIn("factores_positivos", res["xai_explanation"])
        self.assertIn("factores_limitantes", res["xai_explanation"])
        self.assertIn("modelo_base", res["xai_explanation"])
        self.assertIn("precision_modelo", res["xai_explanation"])

    def test_livestock_economics_model(self):
        """Verifica que el modelo económico ganadero calcule la rentabilidad y alertas regulatorias correctamente."""
        from src.models import predict_livestock_economics
        
        res = predict_livestock_economics(
            dept="Antioquia",
            mun="Fredonia",
            species="Bovino",
            purpose="Leche",
            herd_size=30,
            farm_area=40.0,
            producer_type="Mujer Rural (Línea LEC Preferente)"
        )
        
        self.assertIsInstance(res, dict)
        self.assertEqual(res["species"], "Bovino")
        self.assertIn("carrying_capacity_ha", res)
        self.assertIn("max_animals_allowed", res)
        self.assertIn("gross_revenue", res)
        self.assertIn("net_profit", res)
        self.assertIn("margin_pct", res)
        self.assertIn("alerts", res)
        self.assertIn("compliance", res)
        self.assertTrue(res["lec_finagro_activo"])
        self.assertIn("tasa_credito_pct", res)
        
        # Debe calcular ingresos y utilidades reales mayores a cero con hato de 30 bovinos
        self.assertGreater(res["gross_revenue"], 0, "Los ingresos ganaderos proyectados deben ser reales.")
        self.assertIsInstance(res["alerts"], list)

    def test_gel_xml_generation(self):
        """Verifica que se genere un XML válido bajo el estándar GEL-XML del MinTIC."""
        from src.models import generate_gel_xml
        sample_row = {
            "UID": "abc-123",
            "Titulo": "Dataset de Prueba",
            "Descripción": "Descripción larga del dataset.",
            "Información de la Entidad: Nombre de la Entidad": "MinTIC",
            "Información de la Entidad: Sector": "Tecnologías de la Información",
            "Información de la Entidad: Orden": "Nacional",
            "Información de la Entidad: Departamento": "Bogotá",
            "Información de la Entidad: Municipio": "Bogotá D.C.",
            "Número de Filas": 1000,
            "Número de Columnas": 12,
            "ds_score_relevancia": 4.5,
            "ds_calidad_datos": 4.0,
            "url": "https://www.datos.gov.co/abc-123"
        }
        xml_str = generate_gel_xml(sample_row, 0.95)
        self.assertIsInstance(xml_str, str)
        self.assertIn('<?xml version="1.0" encoding="UTF-8"?>', xml_str)
        self.assertIn('<gel:RegistroActivosInformacion', xml_str)
        self.assertIn('<gel:ActivoId>abc-123</gel:ActivoId>', xml_str)
        self.assertIn('<gel:NombreActivo>Dataset de Prueba</gel:NombreActivo>', xml_str)


if __name__ == "__main__":
    unittest.main()
