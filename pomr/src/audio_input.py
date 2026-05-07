"""
Модуль для работы с аудиофайлами: запись, загрузка, преобразование форматов.
"""

import os
from pathlib import Path
from typing import Optional
from pydub import AudioSegment
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import tempfile


class AudioInput:
    """Класс для обработки аудио-ввода."""
    
    SUPPORTED_FORMATS = ('wav', 'mp3', 'm4a', 'flac', 'ogg')
    
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        """
        Инициализация класса.
        
        Args:
            sample_rate: Частота дискретизации (обычно 16000 для Whisper)
            channels: Количество каналов (моно/стерео)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        
        # Проверка доступности микрофона
        try:
            sd.check_input_settings()
            input_name = sd.query_devices("", "input")
            print(f">> Microfon obnaruzhen: {input_name}")
        except Exception as e:
            print(f"WARNING: {e}")
    
    def load_audio(self, audio_path: str) -> AudioSegment:
        """
        Загрузка аудиофайла.
        
        Args:
            audio_path: Путь к аудиофайлу
            
        Returns:
            AudioSegment объект
            
        Raises:
            FileNotFoundError: Если файл не найден
            ValueError: Если формат файла не поддерживается
        """
        path = Path(audio_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {audio_path}")
        
        if path.suffix.lower().lstrip('.') not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Неподдерживаемый формат: {path.suffix}. "
                           f"Поддерживаются: {', '.join(self.SUPPORTED_FORMATS)}")
        
        audio = AudioSegment.from_file(str(path))
        
        # Конвертация в моно, если нужно
        if audio.channels > self.channels:
            audio = audio.set_channels(self.channels)
        
        return audio
    
    def record_mic(self, output_path: Optional[str] = None) -> Path:
        """
        Zapis s mikrofona do nazhatiya Enter.
        
        Returns:
            Path k sohranennomu WAV faylu
        """
        print("\n" + "="*30)
        print("ZAPIS IDET... Govorite zadachi.")
        print(">> Nazhmite ENTER, chtoby ostanovit.")
        print("="*30 + "\n")

        fs = self.sample_rate
        recording = []

        def callback(indata, frames, time, status):
            if status:
                print(status)
            recording.append(indata.copy())

        try:
            with sd.InputStream(samplerate=fs, channels=1, dtype='int16', callback=callback):
                input() # Ozhidaniye nazhatiya Enter
        except Exception as e:
            raise RuntimeError(f"Ne udalos zapustit mikrofon: {e}")

        if not recording:
            raise RuntimeError("Zapis pusta. Mikrofon ne poluchil dannykh.")

        # Sobirayem dannye
        audio_data = np.concatenate(recording, axis=0)

        # Sozdayem put dlya sohraneniya, yesli ego net
        if output_path is None:
            # Sokhranyaem v papku temp_audio vnutri proyekta dlya nadezhnosti
            temp_dir = Path("./temp_audio")
            temp_dir.mkdir(exist_ok=True)
            output_path = temp_dir / "last_record.wav"
        else:
            output_path = Path(output_path)
        
        # Sokhranyaem file na disk
        write(str(output_path), fs, audio_data)
        
        if not output_path.exists():
            raise RuntimeError("File ne byl sohranen na disk!")

        print(f">> Zapis sohranena: {output_path}")
        return output_path
    
    def record_audio(
        self,
        duration_seconds: int = 30,
        output_path: Optional[str] = None,
        format: str = 'wav'
    ) -> Path:
        """
        Запись аудио с микрофона (обёртка над record_mic).
        
        Args:
            duration_seconds: Длительность записи в секундах
            output_path: Путь для сохранения (если None, создаётся временный файл)
            format: Формат выходного файла (wav, mp3 и т.д.)
            
        Returns:
            Path к сохранённому файлу
        """
        # Записываем в WAV (стандарт для Whisper)
        wav_path = self.record_mic(duration_seconds=duration_seconds)
        
        # Если нужен другой формат — конвертируем
        if format.lower() != 'wav':
            return self.convert_format(
                str(wav_path),
                output_format=format,
                output_path=output_path
            )
        
        return wav_path
    
    def convert_format(
        self,
        audio_path: str,
        output_format: str,
        output_path: Optional[str] = None
    ):
        """
        Конвертация аудио в другой формат.
        
        Args:
            audio_path: Путь к исходному файлу
            output_format: Целевой формат (wav, mp3, m4a)
            output_path: Путь для сохранения
            
        Returns:
            Path к конвертированному файлу
        """
        audio = self.load_audio(audio_path)
        
        if output_path is None:
            import tempfile
            stem = Path(audio_path).stem
            output_path = Path(tempfile.mktemp(suffix=f'.{output_format}'))
        
        # Устанавливаем частоту дискретизации для Whisper (16000 Гц)
        audio = audio.set_frame_rate(self.sample_rate)
        
        audio.export(str(output_path), format=output_format.lower())
        
        return output_path
    
    def get_audio_duration(self, audio_path: str) -> float:
        """
        Получение длительности аудиофайла.
        
        Args:
            audio_path: Путь к аудиофайлу
            
        Returns:
            Длительность в секундах
        """
        audio = self.load_audio(audio_path)
        return len(audio) / 1000  # AudioSegment хранит длину в миллисекундах
