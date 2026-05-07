import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class Priority(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

@dataclass
class Task:
    id: int
    text: str
    priority: Priority
    category: str = "другое"

class TaskParser:
    def format_for_output(self, tasks: List[Task]) -> str:
        """Создает реально красивый To-Do List в консоли."""
        if not tasks:
            return "\n⚠️ Задач не обнаружено. Попробуйте продиктовать четче."

        # Сортируем: сначала HIGH, потом MEDIUM, потом LOW
        priority_map = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}
        sorted_tasks = sorted(tasks, key=lambda t: priority_map.get(t.priority, 1))

        lines = [
            "\n" + "═" * 60,
            " 📋 ВАШ СПИСОК ДЕЛ (ОТПОРТИРОВАН ПО ПРИОРИТЕТУ)",
            "═" * 60
        ]

        for task in sorted_tasks:
            icon = {Priority.HIGH: "🔴 СРОЧНО", Priority.MEDIUM: "🟡 НОРМ  ", Priority.LOW: "🟢 ПОЗЖЕ "}.get(task.priority)
            cat_icon = {"работа": "💼", "личное": "🏠", "учеба": "📚", "здоровье": "🏥"}.get(task.category.lower(), "📌")
            
            lines.append(f" {icon} | {cat_icon} {task.text}")

        lines.append("═" * 60 + "\n")
        return "\n".join(lines)
