import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

class OutputManager:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_as_json(self, tasks: list, text: str = "") -> Dict[str, Any]:
        tasks_list = []
        for task in tasks:
            # Если это уже словарь - оставляем, если объект - превращаем в словарь
            if hasattr(task, 'to_dict'):
                tasks_list.append(task.to_dict())
            elif isinstance(task, dict):
                tasks_list.append(task)
            else:
                # На крайний случай ручная сборка
                tasks_list.append({
                    "text": str(getattr(task, 'text', '')),
                    "priority": str(getattr(task, 'priority', 'MEDIUM'))
                })

        return {
            'timestamp': datetime.now().isoformat(),
            'transcription': text,
            'tasks': tasks_list,
            'total_tasks': len(tasks_list),
            'priority_breakdown': self._count_priorities(tasks)
        }

    def _count_priorities(self, tasks: list) -> Dict[str, int]:
        counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for task in tasks:
            # Исправляем обращение: проверяем, объект это или словарь
            if isinstance(task, dict):
                p_val = str(task.get('priority', 'MEDIUM')).upper()
            else:
                # Если это объект Task, берем значение из Enum
                p_val = str(task.priority.value if hasattr(task.priority, 'value') else task.priority).upper()
            
            if p_val in counts:
                counts[p_val] += 1
        return counts

    def save_to_file(self, data: Dict[str, Any], filename: Optional[str] = None) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"tasks_{timestamp}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return filepath

    def print_console(self, tasks: list, text: str = "") -> None:
        # Эту часть теперь лучше вызывать через TaskParser, как мы сделали в main.py
        pass
