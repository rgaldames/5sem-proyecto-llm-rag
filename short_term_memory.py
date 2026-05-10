"""
short_term_memory.py  Memoria de Corto Plazo Conversacional
============================================================
Implementa el componente de memoria de corto plazo del agente usando un buffer
de ventana deslizante que mantiene las ultimas N interacciones en contexto activo.

En LangChain 1.2+, ConversationBufferWindowMemory fue removido de langchain.memory.
Se implementa el buffer directamente con HumanMessage/AIMessage de langchain_core,
que es la forma recomendada en la arquitectura LangGraph moderna.

Arquitectura de Memoria (IL2.2):
+--------------------------------------------------+
|             SISTEMA DE MEMORIA DUAL              |
|                                                  |
|  +-------------------+   +--------------------+  |
|  | CORTO PLAZO       |   | LARGO PLAZO        |  |
|  | Buffer deslizante |   | MemoryStore (JSON+ |  |
|  | HumanMessage /    |   | Embeddings coseno) |  |
|  | AIMessage (k=5)   |   | memory_store.py    |  |
|  +-------------------+   +--------------------+  |
+--------------------------------------------------+

- Corto plazo : coherencia inmediata de la conversacion activa
- Largo plazo : recuperacion semantica de sesiones anteriores
"""

from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


class ShortTermMemory:
    """
    Buffer de ventana deslizante implementado con mensajes de LangChain Core.

    Mantiene las ultimas `window_size` interacciones (pares humano/AI) como
    objetos HumanMessage/AIMessage, que son directamente compatibles con el
    nuevo create_agent de LangChain 1.2+.
    """

    def __init__(self, window_size: int = 5):
        """
        Inicializa la memoria de corto plazo.

        Args:
            window_size: Numero de intercambios a conservar (cada intercambio =
                         1 HumanMessage + 1 AIMessage). Default=5.
        """
        self.window_size = window_size
        self._messages: List[BaseMessage] = []
        self._turn_count: int = 0

    def save_turn(self, human_input: str, ai_output: str) -> None:
        """
        Guarda un turno completo de conversacion.

        Aplica la ventana deslizante: si el buffer supera window_size*2 mensajes
        (pares humano+AI), elimina el par mas antiguo.

        Args:
            human_input: El mensaje que envio el usuario.
            ai_output: La respuesta generada por el agente.
        """
        self._messages.append(HumanMessage(content=human_input))
        self._messages.append(AIMessage(content=ai_output))
        self._turn_count += 1

        # Ventana deslizante: mantener solo los ultimos window_size pares
        max_messages = self.window_size * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    def get_messages(self) -> List[BaseMessage]:
        """
        Retorna los mensajes actuales del buffer.

        Returns:
            Lista de objetos HumanMessage/AIMessage de LangChain Core.
            Compatible directamente con create_agent de LangChain 1.2+.
        """
        return list(self._messages)

    def get_history_as_text(self) -> str:
        """
        Retorna el historial de conversacion como texto plano legible.
        Util para logging, depuracion o inyeccion en prompts de texto.

        Returns:
            String formateado con la conversacion reciente.
        """
        if not self._messages:
            return ""

        lines = []
        for msg in self._messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"Usuario: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Asistente: {msg.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Limpia el buffer (nueva sesion)."""
        self._messages = []
        self._turn_count = 0

    @property
    def turn_count(self) -> int:
        """Numero de turnos guardados en la sesion actual."""
        return self._turn_count

    @property
    def is_empty(self) -> bool:
        """True si no hay historial conversacional aun."""
        return len(self._messages) == 0

    def get_summary_stats(self) -> Dict:
        """
        Estadisticas del estado actual de la memoria.

        Returns:
            Diccionario con metricas de uso.
        """
        max_msgs = self.window_size * 2
        return {
            "window_size": self.window_size,
            "total_turns_session": self._turn_count,
            "messages_in_buffer": len(self._messages),
            "buffer_full": len(self._messages) >= max_msgs,
        }
