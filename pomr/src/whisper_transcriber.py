"""
Модуль для транскрибации аудио в текст с помощью Whisper.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor
import torch


class WhisperTranscriber:
    """Класс для транскрибации аудио через Whisper."""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: Optional[str] = None,
        language: str = "ru",
        task: str = "transcribe"
    ):
        """
        Инициализация транскрибера.
        
        Args:
            model_name: Название модели Whisper (small, medium, large)
            device: Устройство для inference ('cuda', 'cpu' или None для автовыбора)
            language: Язык транскрипции
            task: Тип задачи ('transcribe' или 'translate')
        """
        self.model_name = model_name
        self.language = language
        self.task = task
        
        # Автовыбор устройства
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        print(f"Загрузка модели Whisper: {model_name} на устройстве {device}")
        
        # Загрузка модели и процессора
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        if self.device == "cuda":
            self.model = self.model.half()
        
        self.model.to(self.device)
        
        # Загрузка токенизатора
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("Whisper транскрибер инициализирован успешно!")
    
    def transcribe(
        self,
        audio_path: str,
        chunk_size: int = 30,
        chunk_overlap: int = 10
    ) -> Dict[str, Any]:
        """
        Транскрибация аудиофайла в текст.
        
        Args:
            audio_path: Путь к аудиофайлу
            chunk_size: Размер chunks для обработки (в секундах)
            chunk_overlap: Пересечение между chunks (в секундах)
            
        Returns:
            Словарь с результатами транскрипции:
            - text: Полный текст
            - segments: Список сегментов с таймкодами
            - duration: Длительность аудио
        """
        from src.audio_input import AudioInput
        
        # Загрузка аудио
        audio_input = AudioInput(sample_rate=16000)
        audio = audio_input.load_audio(audio_path)
        
        print(f"Длительность аудио: {len(audio)/1000:.2f} сек")
        
        # Обработка в chunks для больших файлов
        if len(audio) > chunk_size * 1000:
            text_parts = []
            segments = []
            
            start_time = 0
            num_chunks = int(len(audio) / (chunk_size * 1000))
            
            for i in range(num_chunks):
                start = start_time * 1000
                end = min((start + chunk_size) * 1000, len(audio))
                
                chunk = audio[start:end]
                
                # Konvertiruyem AudioSegment v massiv chisel pered obrabotkoy
                chunk_array = self._preprocess_audio(chunk)
                result = self._transcribe_chunk(chunk_array)
                text_parts.append(result['text'])
                segments.extend(result['segments'])
                
                start_time += chunk_size
            
            full_text = ' '.join(text_parts)
        else:
            # Конвертируем AudioSegment в массив чисел перед обработкой
            audio_array = self._preprocess_audio(audio)
            result = self._transcribe_chunk(audio_array)
            full_text = result['text']
            segments = result.get('segments', [])  # Используем .get для безопасности
        
        return {
            'text': full_text,
            'segments': segments,
            'duration': len(audio) / 1000
        }
    
    def _preprocess_audio(self, audio_segment) -> np.ndarray:
        """
        Преобразование AudioSegment в нормализованный numpy массив для Whisper.
        
        Args:
            audio_segment: AudioSegment объект из pydub
            
        Returns:
            Нормализованный numpy массив (float32, диапазон [-1, 1])
        """
        # Получаем сырые сэмплы
        samples = audio_segment.get_array_of_samples()
        
        # Конвертируем в numpy
        audio_array = np.array(samples, dtype=np.int16)
        
        # Нормализуем в диапазон [-1, 1] (Whisper ожидает float32 в этом диапазоне)
        audio_float = audio_array.astype(np.float32) / 32768.0
        
        # Усиление звука до максимума (нормализация)
        if np.max(np.abs(audio_float)) > 0:
            audio_float = audio_float / np.max(np.abs(audio_float))
        
        # Если стерео — конвертируем в моно
        if audio_segment.channels == 2:
            audio_float = np.mean(audio_float, axis=1)
        
        return audio_float
    
    def _transcribe_chunk(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """
        Транскрибация одного аудио chunk.
        
        Args:
            audio_array: Аудио chunk в формате numpy array (float32, [-1, 1])
            
        Returns:
            Результат транскрипции
        """
        # Подготовка ввода для Whisper
        inputs = self.processor(
            audio=audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            transcript = self.model.generate(
                **inputs,
                language=self.language,
                task=self.task
            )
        
        # Декодирование текста
        text = self.processor.batch_decode(transcript, skip_special_tokens=True)[0]
        
        return {
            'text': text,
            'segments': []  # Добавляем пустой список, чтобы не было ошибки
        }
    
    def transcribe_from_audio_input(
        self,
        audio_path: str,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Транскрибация аудиофайла через AudioInput.
        
        Args:
            audio_path: Путь к аудиофайлу
            sample_rate: Частота дискретизации
            
        Returns:
            Результат транскрипции
        """
        from src.audio_input import AudioInput
        
        # Загрузка аудио
        audio_input = AudioInput(sample_rate=sample_rate)
        audio_segment = audio_input.load_audio(audio_path)
        
        print(f"Длительность аудио: {len(audio_segment)/1000:.2f} сек")
        
        # Конвертация AudioSegment -> numpy (препроцессинг для Whisper)
        audio_array = self._preprocess_audio(audio_segment)
        
        return self._transcribe_chunk(audio_array)
