"""
tools.py  Herramientas del Agente Veterinario
=============================================
Define las tres categorías de herramientas que el agente ReAct puede invocar:
  1. CONSULTA   search_clinical_db: busca en ChromaDB historiales clínicos
  2. ESCRITURA  write_visit_summary: genera y guarda un resumen de consulta
  3. RAZONAMIENTO  analyze_symptoms: razona sobre síntomas y decide si es urgente

Cada herramienta es un LangChain StructuredTool (usa @tool decorator).
"""

import os
import json
import time
import uuid
from langchain_core.tools import tool


# --------------------------------------------------
# HERRAMIENTA 1: CONSULTA (IE1  Herramienta RAG)
# --------------------------------------------------
def make_search_tool(vectorstore):
    """Fábrica: retorna la herramienta de búsqueda enlazada al vectorstore activo."""

    @tool
    def search_clinical_db(query: str) -> str:
        """
        Busca información clínica veterinaria relevante en la base de datos interna
        de la clínica (ChromaDB). Útil para responder preguntas sobre razas, tratamientos
        previos, diagnósticos registrados o medicamentos usados en pacientes anteriores.

        Args:
            query: Pregunta o tema a buscar en los historiales clínicos.

        Returns:
            Fragmentos de historiales clínicos relacionados con la consulta.
        """
        try:
            docs = vectorstore.similarity_search(query, k=3)
            if not docs:
                return "No se encontraron historiales clínicos relevantes para esa consulta."
            result = []
            for i, doc in enumerate(docs, 1):
                result.append(f"[Registro #{i}]\n{doc.page_content}")
            return "\n\n".join(result)
        except Exception as e:
            return f"Error al consultar la base de datos: {str(e)}"

    return search_clinical_db


# ----------------------------------------------------------
# HERRAMIENTA 2: ESCRITURA (IE1  Herramienta de Escritura)
# ----------------------------------------------------------
SUMMARIES_PATH = "./visit_summaries.jsonl"

@tool
def write_visit_summary(patient_name: str, species: str, concern: str, advice_given: str) -> str:
    """
    Genera y guarda un resumen estructurado de la consulta virtual con el dueño de la mascota.
    Úsala SIEMPRE al final de una interacción para dejar registro permanente de lo conversado.
    El resumen queda guardado en 'visit_summaries.jsonl' para auditoría futura.

    Args:
        patient_name: Nombre de la mascota según lo indicó el dueño.
        species: Especie de la mascota (perro, gato, ave, etc.).
        concern: Problema o motivo principal de consulta descrito por el dueño.
        advice_given: Resumen del consejo o información entregada al dueño.

    Returns:
        Confirmación de que el resumen fue guardado correctamente.
    """
    try:
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_unix": int(time.time()),
            "patient_name": patient_name,
            "species": species,
            "concern": concern,
            "advice_given": advice_given,
        }

        with open(SUMMARIES_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return (
            f"[OK] Resumen guardado exitosamente para {patient_name} ({species}).\n"
            f"ID de registro: {record['id']}\n"
            f"Fecha: {record['timestamp']}"
        )
    except Exception as e:
        return f"[ERROR] Error al guardar el resumen: {str(e)}"


# ----------------------------------------------------------------------
# HERRAMIENTA 3: RAZONAMIENTO (IE1  Herramienta de Toma de Decisiones)
# ----------------------------------------------------------------------
# Síntomas que indican urgencia veterinaria inmediata
URGENT_SYMPTOMS = [
    "convulsión", "convulsiones", "desmayo", "pérdida de consciencia",
    "sangrado excesivo", "hemorragia", "no respira", "dificultad respiratoria",
    "vómitos con sangre", "diarrea con sangre", "parálisis", "mordedura de serpiente",
    "intoxicación", "envenenamiento", "fractura", "trauma", "atropellado",
    "no puede caminar", "colapso", "inconsciencia", "ojos amarillos",
    "no orina", "obstrucción urinaria", "distensión abdominal severa",
]

MODERATE_SYMPTOMS = [
    "vómito", "diarrea", "letargo", "sin apetito", "no come",
    "cojea", "rascado excesivo", "pérdida de pelo", "tos",
    "estornudos", "secreción nasal", "secreción ocular", "fiebre",
    "pérdida de peso", "sed excesiva", "orina excesiva",
]

@tool
def analyze_symptoms(symptoms_description: str) -> str:
    """
    Analiza una descripción de síntomas para determinar el nivel de urgencia
    y decide qué tipo de respuesta o acción recomendar al dueño de la mascota.
    Esta herramienta implementa razonamiento basado en reglas para planificación adaptativa.

    Niveles de urgencia:
    - CRÍTICO: Requiere atención veterinaria de emergencia INMEDIATA.
    - MODERADO: Requiere visita veterinaria pronto (máximo 24-48 horas).
    - LEVE: Puede monitorearse en casa con cuidados básicos.

    Args:
        symptoms_description: Descripción en texto libre de los síntomas que presenta la mascota.

    Returns:
        Análisis del nivel de urgencia y recomendaciones de acción concretas.
    """
    desc_lower = symptoms_description.lower()

    # Regla 1: Detección de síntomas críticos
    found_urgent = [s for s in URGENT_SYMPTOMS if s in desc_lower]
    if found_urgent:
        return (
            f"[CRITICO] NIVEL DE URGENCIA: CRÍTICO\n"
            f"Síntomas detectados que requieren atención inmediata: {', '.join(found_urgent)}.\n\n"
            f"ACCIÓN REQUERIDA: Llevar a la mascota a una clínica veterinaria de emergencias "
            f"AHORA MISMO. No esperes. Mientras trasladas al animal, mantén la calma, "
            f"no le des medicamentos sin indicación veterinaria, y si sangra, aplica presión "
            f"suave con un paño limpio.\n\n"
            f"[AVISO] Este bot NO puede reemplazar atención médica de emergencia."
        )

    # Regla 2: Detección de síntomas moderados
    found_moderate = [s for s in MODERATE_SYMPTOMS if s in desc_lower]
    if found_moderate:
        count = len(found_moderate)
        priority = "ALTA" if count >= 3 else "MODERADA"
        return (
            f"[AVISO] NIVEL DE URGENCIA: {priority}\n"
            f"Síntomas identificados: {', '.join(found_moderate)}.\n\n"
            f"PLAN RECOMENDADO:\n"
            f"1. Monitorea al animal de cerca las próximas 12-24 horas.\n"
            f"2. Asegúrate de que tenga acceso a agua fresca.\n"
            f"3. Ofrece comida blanda y en porciones pequeñas si tiene problemas digestivos.\n"
            f"4. Agenda una cita veterinaria dentro de las próximas 24-48 horas.\n"
            f"5. Si los síntomas empeoran, acude a urgencias.\n\n"
            f" Puedo buscar información clínica relevante en nuestra base de datos si necesitas más detalles."
        )

    # Regla 3: Sin síntomas de alerta detectados
    return (
        f"[OK] NIVEL DE URGENCIA: LEVE / SIN ALERTA INMEDIATA\n"
        f"No se detectaron síntomas de alta urgencia en la descripción proporcionada.\n\n"
        f"RECOMENDACIONES GENERALES:\n"
        f"1. Mantén la rutina de alimentación y ejercicio habitual.\n"
        f"2. Asegura que las vacunas y desparasitaciones estén al día.\n"
        f"3. Si tienes dudas específicas, puedo consultar la base de datos clínica de la veterinaria.\n"
        f"4. Agenda chequeos preventivos cada 6-12 meses.\n\n"
        f" ¿Hay algo específico en lo que pueda orientarte hoy?"
    )
