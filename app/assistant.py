import os
import re
import pandas as pd
import numpy as np

import requests

class CampoAssistant:
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.df_soils = self._load_csv("upra_aptitud_suelos.csv", is_processed=False)
        self.df_weather = self._load_csv("ideam_clima_historico.csv", is_processed=False)
        self.df_production = self._load_csv("produccion_historica.csv", is_processed=False)
        self.df_credits = self._load_csv("finagro_creditos.csv", is_processed=False)
        self.df_prices = self._load_csv("sipsa_precios_mercado.csv", is_processed=False)
        self.df_legal = self._load_csv("seguridad_juridica_tierras.csv", is_processed=False)
        self.df_livestock = self._load_csv("inventario_ganadero_nacional.csv", is_processed=False)

    def _load_csv(self, filename, is_processed=False):
        folder = self.processed_dir if is_processed else self.raw_dir
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            try:
                return pd.read_csv(path, encoding="utf-8")
            except Exception as e:
                print(f"[ASSISTANT] Error al cargar {filename}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def search_context(self, query):
        """
        Busca información relevante en los conjuntos de datos locales basados en la consulta.
        """
        query_lower = query.lower()
        context_parts = []

        # 1. Identificar Departamento y Municipio
        detected_dept = None
        detected_mun = None
        
        # Lista de departamentos comunes
        depts = ["antioquia", "boyaca", "boyacá", "caldas", "cauca", "cesar", "cordoba", "córdoba", "cundinamarca", "huila", "meta", "nariño", "narino", "risaralda", "santander", "sucre", "tolima", "valle"]
        for d in depts:
            if d in query_lower:
                detected_dept = d
                break

        # Lista de municipios comunes en nuestros mock data
        muns = ["fredonia", "jerico", "jericó", "andes", "espinal", "ibague", "ibagué", "purificacion", "purificación", "pitalito", "garzon", "garzón", "neiva", "granada", "villavicencio", "acacias", "acacías", "cerete", "cereté", "monteria", "montería", "sahagun", "sahagún", "pasto", "ipiales", "tumaco", "tunja", "duitama", "sogamoso", "fusagasuga", "fusagasugá", "girardot", "facatativa", "facatativá", "popayan", "popayán", "quilichao", "miranda", "valledupar", "aguachica"]
        for m in muns:
            if m in query_lower:
                detected_mun = m
                break

        # 2. Identificar Cultivo o Ganado
        detected_crop = None
        crops = ["cafe", "café", "cacao", "arroz", "maiz", "maíz", "platano", "plátano"]
        for c in crops:
            if c in query_lower:
                detected_crop = c
                break

        # 3. Identificar si pregunta por precios, créditos, tierras o clima
        ask_price = any(w in query_lower for w in ["precio", "valor", "costar", "vender", "sipsa", "mercado", "kilo", "tonelada"])
        ask_credit = any(w in query_lower for w in ["credito", "crédito", "finagro", "banco", "tasa", "interes", "financiación", "financiar"])
        ask_legal = any(w in query_lower for w in ["tierra", "formalizar", "titulo", "título", "escritura", "ant", "catastro", "seguridad jurídica", "restitución"])
        ask_weather = any(w in query_lower for w in ["clima", "temperatura", "lluvia", "precipitacion", "ideam", "mes", "siembra", "sembrar"])
        ask_livestock = any(w in query_lower for w in ["ganado", "vaca", "cerdo", "bovino", "porcino", "poblacion", "aftosa", "ica"])

        # ── EXTRACCIÓN DE CONTEXTO POR TEMA ──
        
        # A. Precios SIPSA
        if ask_price and not self.df_prices.empty:
            prices_subset = self.df_prices.copy()
            if detected_crop:
                # Mapear nombre normalizado
                crop_map = {"cafe": "Cafe", "café": "Cafe", "cacao": "Cacao", "arroz": "Arroz", "maiz": "Maiz", "maíz": "Maiz", "platano": "Platano", "plátano": "Platano"}
                crop_name = crop_map.get(detected_crop, "Cafe")
                prices_subset = prices_subset[prices_subset["cultivo"] == crop_name]
            
            if not prices_subset.empty:
                latest = prices_subset.sort_values("fecha").tail(5)
                context_parts.append("💰 **Últimos precios de mercado reportados (SIPSA):**")
                for _, r in latest.iterrows():
                    context_parts.append(f"- {r['cultivo']}: Promedio de ${int(r['precio_kg_promedio_cop']):,} COP/kg en la central {r['central_abasto']} (Fecha: {r['fecha']})")

        # B. Producción Histórica EVA y Aptitud UPRA
        if detected_crop and not self.df_production.empty:
            crop_map = {"cafe": "Cafe", "café": "Cafe", "cacao": "Cacao", "arroz": "Arroz", "maiz": "Maiz", "maíz": "Maiz", "platano": "Platano", "plátano": "Platano"}
            crop_name = crop_map.get(detected_crop, "Cafe")
            
            prod_subset = self.df_production[self.df_production["cultivo"] == crop_name]
            if detected_dept:
                prod_subset = prod_subset[prod_subset["departamento"].str.lower().str.contains(detected_dept)]
            
            if not prod_subset.empty:
                avg_yield = prod_subset["rendimiento_ton_ha"].mean()
                total_ha = prod_subset["hectareas_sembradas"].sum()
                context_parts.append(f"🌾 **Datos de producción histórica (EVA) para {crop_name}:**")
                context_parts.append(f"- Rendimiento promedio en la zona: {avg_yield:.2f} Toneladas por Hectárea.")
                context_parts.append(f"- Hectáreas sembradas históricamente registradas: {total_ha:,.1f} Ha.")
                
            # Aptitud de suelos
            if not self.df_soils.empty:
                soil_sub = self.df_soils[self.df_soils["cultivo"] == crop_name]
                if detected_dept:
                    soil_sub = soil_sub[soil_sub["departamento"].str.lower().str.contains(detected_dept)]
                if not soil_sub.empty:
                    aptitud = soil_sub["aptitud"].mode()[0] if not soil_sub["aptitud"].empty else "Media"
                    ph = soil_sub["ph_suelo"].mean()
                    alt = soil_sub["altitud_m"].mean()
                    context_parts.append(f"🌱 **Criterios de Suelo (UPRA) para {crop_name}:**")
                    context_parts.append(f"- Aptitud promedio del suelo en la zona: {aptitud}")
                    context_parts.append(f"- pH promedio registrado: {ph:.2f} | Altitud media: {int(alt)} m.s.n.m.")

        # C. Créditos Finagro
        if ask_credit and not self.df_credits.empty:
            cred_sub = self.df_credits.copy()
            if detected_dept:
                cred_sub = cred_sub[cred_sub["departamento"].str.lower().str.contains(detected_dept)]
            if detected_crop:
                crop_map = {"cafe": "Cafe", "café": "Cafe", "cacao": "Cacao", "arroz": "Arroz", "maiz": "Maiz", "maíz": "Maiz", "platano": "Platano", "plátano": "Platano"}
                crop_name = crop_map.get(detected_crop, "Cafe")
                cred_sub = cred_sub[cred_sub["cultivo"] == crop_name]
                
            if not cred_sub.empty:
                monto = cred_sub["monto_creditos_cop"].sum()
                tasa = cred_sub["tasa_interes_promedio"].mean()
                context_parts.append("🏦 **Información de financiamiento rural (FINAGRO):**")
                context_parts.append(f"- Monto total aprobado en registros: ${int(monto):,} COP.")
                context_parts.append(f"- Tasa de interés promedio ponderada de la zona: {tasa:.2f}% E.A.")

        # D. Clima IDEAM
        if ask_weather and not self.df_weather.empty:
            weather_sub = self.df_weather.copy()
            if detected_dept:
                weather_sub = weather_sub[weather_sub["departamento"].str.lower().str.contains(detected_dept)]
            if not weather_sub.empty:
                latest_weather = weather_sub.tail(5)
                temp = weather_sub["temperatura_promedio_c"].mean()
                rain = weather_sub["precipitacion_diaria_mm"].sum() / (len(weather_sub) / 365) if len(weather_sub) > 0 else 1800
                context_parts.append("🌦️ **Monitoreo meteorológico e histórico (IDEAM):**")
                context_parts.append(f"- Temperatura promedio histórica anual: {temp:.1f} °C.")
                context_parts.append(f"- Precipitación promedio acumulada anual estimada: {int(rain)} mm.")

        # E. Seguridad Jurídica de la Tierra
        if ask_legal and not self.df_legal.empty:
            legal_sub = self.df_legal.copy()
            if detected_dept:
                legal_sub = legal_sub[legal_sub["departamento"].str.lower().str.contains(detected_dept)]
            if not legal_sub.empty:
                isj = legal_sub["indice_seguridad_juridica"].mean()
                cat = legal_sub["hectareas_catastradas"].sum()
                form = legal_sub["hectareas_formalizadas"].sum()
                context_parts.append("⚖️ **Estado catastral y formalización de tierras (ANT/UPRA):**")
                context_parts.append(f"- Índice de Seguridad Jurídica de la Tierra: {isj:.1f}%")
                context_parts.append(f"- Hectáreas catastradas: {cat:,.1f} Ha | Hectáreas formalizadas: {form:,.1f} Ha (Porcentaje formalización: {(form/cat*100):.1f}%)." if cat > 0 else "")

        # F. Ganadería ICA
        if ask_livestock and not self.df_livestock.empty:
            live_sub = self.df_livestock.copy()
            if detected_dept:
                live_sub = live_sub[live_sub["departamento"].str.lower().str.contains(detected_dept)]
            if not live_sub.empty:
                bovino = live_sub["poblacion_bovina"].mean()
                porcino = live_sub["poblacion_porcina"].mean()
                aftosa = live_sub["vacunacion_fiebre_aftosa_pct"].mean()
                context_parts.append("🐄 **Cifras del inventario ganadero y sanidad (ICA):**")
                context_parts.append(f"- Población bovina promedio por municipio: {int(bovino):,} cabezas.")
                context_parts.append(f"- Población porcina promedio: {int(porcino):,} cabezas.")
                context_parts.append(f"- Cobertura de vacunación aftosa: {aftosa:.2f}% (Meta recomendada > 95%).")

        # G. Enfoque Territorial Priorizado (Estrategia 3)
        if any(z in query_lower for z in ["caqueta", "caquetá", "guaviare", "amazonía", "amazonia", "orinoquía", "orinoquia", "putumayo", "sustainability", "sostenible", "deforestacion", "deforestación"]):
            context_parts.append("🌳 **Prioridad de Sostenibilidad y Frontera Agrícola (Piloto Piedemonte/Amazonía):**")
            context_parts.append("- Esta zona tiene un alto índice de riesgo de deforestación por expansión agropecuaria no planificada.")
            context_parts.append("- C.A.M.P.O. en esta región prioriza la estabilización de la frontera agrícola, protegiendo los Resguardos Indígenas y fomentando la agroforestería (sistemas de cacao y café bajo sombra) y sistemas silvopastoriles para ganadería sostenible.")
            context_parts.append("- Se aplican regulaciones ambientales CAR (Decreto 1076) e incentivos del Gobierno Nacional (PDET) para la sustitución de economías no sostenibles y prevención de la tala de bosques primarios.")

        return "\n".join(context_parts)

    def answer_question(self, query, api_key=None):
        """
        Responde la consulta del campesino usando RAG.
        Usa la API de OpenRouter con el modelo meta-llama/llama-3.1-8b-instruct.
        """
        # 1. Obtener contexto
        context = self.search_context(query)
        
        # 2. Clave de OpenRouter
        # Usamos la clave proporcionada en el contexto anterior para que funcione de manera silenciosa
        openrouter_key = "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"
        
        prompt = f"""
Eres AgroIA, un asistente de Inteligencia Artificial para el campo colombiano que ayuda a los pequeños y medianos campesinos.
Habla en un lenguaje amable, claro, sencillo, muy respetuoso ("Sumercé", "Don", "Doña") y con terminología rural colombiana, pero manteniendo el rigor científico.

El usuario te hace la siguiente pregunta: "{query}"

Usa la siguiente información real extraída de las bases de datos gubernamentales (UPRA, ICA, IDEAM, FINAGRO, SIPSA) como tu única fuente de verdad para el contexto técnico de la respuesta:
---
{context if context else "No se encontraron registros específicos en la base de datos para esta consulta."}
---

Instrucciones:
1. Responde a la pregunta del campesino de manera directa, práctica y fácil de entender.
2. Explícale con paciencia los términos técnicos (como pH del suelo, altitud, ROI, o tasas de interés) si es necesario.
3. Si el contexto incluye datos, cítalos para respaldar tu respuesta (ejemplo: precios de SIPSA, rendimientos de EVA, créditos de FINAGRO o áreas de la ANT).
4. Dale consejos agrícolas prácticos para mejorar el rendimiento de su cosecha o finca ganadera según lo aprendido del contexto.
5. Si no hay datos específicos en el contexto, sé honesto y dale consejos generales basados en tu conocimiento agronómico para predios en Colombia.
"""

        try:
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "meta-llama/llama-3.1-8b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "top_p": 0.9,
            }
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data, timeout=15)
            response.raise_for_status()
            res_json = response.json()
            return res_json['choices'][0]['message']['content']
        except Exception as e:
            print(f"[ASSISTANT] Error en OpenRouter API: {e}. Activando fallback inteligente...")
            return self._generate_fallback_response(query, context, error_msg=str(e))

    def _generate_fallback_response(self, query, context, error_msg=None):
        """
        Genera una respuesta inteligente estructurada basada en los datos encontrados.
        Se activa cuando no hay API Key de Gemini o falla la llamada.
        """
        intro = """¡Hola, mi querido productor! Soy **AgroIA**, su copiloto analítico del campo. 

Como no tengo conexión activa a internet o no se ha configurado la clave de la API de Gemini, le responderé directamente usando las bases de datos locales del sistema. Aquí tiene un reporte detallado según lo que me preguntó:

"""
        if error_msg:
            intro = f"""*Nota: Se detectó un detalle técnico con la API ({error_msg[:60]}...), por lo que activamos el sistema local.* \n\n""" + intro

        if not context:
            return intro + """He revisado las bases de datos de **UPRA, IDEAM, ICA, FINAGRO y SIPSA**, pero no encontré datos específicos de municipios o cultivos en su consulta.

**Recomendaciones Generales para su Finca:**
1. **Análisis de Suelo:** Antes de sembrar, le sugerimos hacer una muestra de tierra. El café y cacao prefieren suelos ligeramente ácidos (pH de 5.5 a 6.5). El arroz y maíz toleran rangos más amplios.
2. **Épocas de Lluvia:** Recuerde planear la siembra para el inicio de las temporadas de lluvias locales (comúnmente abril-mayo y octubre-noviembre) para no depender tanto de riego artificial.
3. **Financiación:** FINAGRO ofrece líneas de crédito de fomento para pequeños productores con tasas muy favorables a través de su banco de confianza (como el Banco Agrario).

*Pregúnteme por un departamento o cultivo específico (ejemplo: "precios de café en Antioquia") para poder darle datos exactos de nuestras bases.*"""

        # Estructurar la información de contexto de manera legible
        sections = context.split("\n\n")
        response_body = "Basado en los microdatos oficiales que he analizado:\n\n"
        for sec in sections:
            response_body += sec + "\n\n"

        response_body += """**Consejo práctico de AgroIA:**
- Si va a sembrar, asegúrese de que el pH de su suelo sea adecuado.
- Monitoree los precios de centrales de abasto (SIPSA) semanales en el dashboard para vender en el mejor momento.
- Para créditos, acérquese al Banco Agrario y solicite las líneas especiales de tasa subsidiada de FINAGRO para Pequeños Productores.

¿Tiene alguna otra duda sobre su predio o cultivo? Escríbame indicando el municipio o el producto y con gusto le ayudo."""

        return intro + response_body
