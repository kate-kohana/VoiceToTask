"""
Тесты для VoiceToTaskApp pipeline.
"""

import unittest
from pathlib import Path
import sys

# Добавляем src в путь для импорта
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from audio_input import AudioInput
from task_parser import TaskParser, Priority
from output_manager import OutputManager


class TestAudioInput(unittest.TestCase):
    """Тесты для AudioInput."""
    
    def setUp(self):
        self.audio_input = AudioInput(sample_rate=16000)
    
    def test_supported_formats(self):
        """Проверка поддерживаемых форматов."""
        self.assertIn('wav', self.audio_input.SUPPORTED_FORMATS)
        self.assertIn('mp3', self.audio_input.SUPPORTED_FORMATS)
    
    def test_get_audio_duration(self):
        """Проверка получения длительности (демо)."""
        # Создаём тестовый аудиофайл
        from pydub import AudioSegment
        audio = AudioSegment.silent(duration=1000, sample_width=2, channel_width=1)
        audio_path = Path("./test_audio.wav")
        audio.export(str(audio_path), format="wav")
        
        duration = self.audio_input.get_audio_duration(str(audio_path))
        self.assertAlmostEqual(duration, 1.0, places=1)
        
        # Очистка
        audio_path.unlink()


class TestTaskParser(unittest.TestCase):
    """Тесты для TaskParser."""
    
    def setUp(self):
        self.parser = TaskParser()
    
    def test_parse_simple_task(self):
        """Парсинг простой задачи."""
        text = "Купить молоко"
        tasks = self.parser.parse_tasks(text)
        
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].text, "Купить молоко")
        self.assertIn(tasks[0].priority, [Priority.MEDIUM, Priority.LOW])
    
    def test_parse_multiple_tasks(self):
        """Парсинг нескольких задач."""
        text = "Купить молоко и позвонить маме"
        tasks = self.parser.parse_tasks(text)
        
        self.assertEqual(len(tasks), 2)
    
    def test_determine_priority_high(self):
        """Определение HIGH приоритета."""
        text = "Срочно сделать отчет к завтрашнему дню"
        tasks = self.parser.parse_tasks(text)
        
        self.assertEqual(tasks[0].priority, Priority.HIGH)
    
    def test_determine_priority_medium(self):
        """Определение MEDIUM приоритета."""
        text = "Купить продукты"
        tasks = self.parser.parse_tasks(text)
        
        self.assertEqual(tasks[0].priority, Priority.MEDIUM)
    
    def test_determine_priority_low(self):
        """Определение LOW приоритета."""
        text = "Можно вынести мусор когда-нибудь"
        tasks = self.parser.parse_tasks(text)
        
        self.assertEqual(tasks[0].priority, Priority.LOW)
    
    def test_sort_by_priority(self):
        """Сортировка по приоритету."""
        tasks = [
            TaskParser.Task(id=1, text="Купить молоко", priority=Priority.MEDIUM),
            TaskParser.Task(id=2, text="Срочно сделать отчет", priority=Priority.HIGH),
            TaskParser.Task(id=3, text="Вынести мусор", priority=Priority.LOW),
        ]
        
        sorted_tasks = self.parser.sort_by_priority(tasks)
        
        self.assertEqual(sorted_tasks[0].priority, Priority.HIGH)
        self.assertEqual(sorted_tasks[1].priority, Priority.MEDIUM)
        self.assertEqual(sorted_tasks[2].priority, Priority.LOW)
    
    def test_group_by_category(self):
        """Группировка по категориям."""
        tasks = [
            TaskParser.Task(id=1, text="Сделать отчет", priority=Priority.HIGH, category='работа'),
            TaskParser.Task(id=2, text="Купить продукты", priority=Priority.MEDIUM, category='личное'),
        ]
        
        groups = self.parser.group_by_category(tasks)
        
        self.assertIn('работа', groups)
        self.assertIn('личное', groups)


class TestOutputManager(unittest.TestCase):
    """Тесты для OutputManager."""
    
    def setUp(self):
        self.output = OutputManager(output_dir="./test_output")
    
    def tearDown(self):
        import shutil
        if Path("./test_output").exists():
            shutil.rmtree("./test_output")
    
    def test_format_as_json(self):
        """Форматирование в JSON."""
        tasks = [
            {'id': 1, 'text': 'Купить молоко', 'priority': 'MEDIUM'},
            {'id': 2, 'text': 'Срочно сделать отчет', 'priority': 'HIGH'},
        ]
        
        data = self.output.format_as_json(tasks, "Текст заметки")
        
        self.assertIn('tasks', data)
        self.assertEqual(data['total_tasks'], 2)
        self.assertIn('priority_breakdown', data)


if __name__ == '__main__':
    unittest.main()
