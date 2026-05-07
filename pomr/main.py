"""
VoiceToTaskApp - Preobrazovanie golosovyh zametok v spisok zadach s prioritetami

Osnovnoy file prilozheniya (tochka vvoda).
"""

import sys
import os
from pathlib import Path

# Dobavlyaem src v put dlya import
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.audio_input import AudioInput
from src.whisper_transcriber import WhisperTranscriber
from src.llm_analyzer import TaskAnalyzer
from src.task_parser import TaskParser
from src.output_manager import OutputManager


def create_pipeline():
    """
    Sozdanie pipeline prilozheniya.
    
    Returns:
        Slovar s inicializirovannymi komponentami
    """
    # Inicializaciya audio vvoda
    audio_input = AudioInput(sample_rate=16000)
    
    # Inicializaciya Whisper transcribera
    whisper_transcriber = WhisperTranscriber(
        model_name="openai/whisper-medium",
        device=None,  # auto (cuda ili cpu)
        language="ru"
    )
    
    # Inicializaciya LLM analytika (Qwen 2.5)
    llm_analyzer = TaskAnalyzer()
    
    # Inicializaciya parsera zadach
    task_parser = TaskParser()
    
    # Inicializaciya menedzhera vyvoda
    output_manager = OutputManager(output_dir="./output")
    
    return {
        'audio_input': audio_input,
        'whisper': whisper_transcriber,
        'llm': llm_analyzer,
        'parser': task_parser,
        'output': output_manager
    }


def process_audio_file(audio_path: str, pipeline):
    print("\n" + "─" * 40)
    print("⏳ Обработка... Пожалуйста, подождите.")
    
    # 1. Транскрибация
    transcription = pipeline['whisper'].transcribe(audio_path)
    
    # 2. Анализ ИИ
    analysis = pipeline['llm'].analyze_text(transcription['text'])
    
    # 3. Превращаем "сырой" JSON от ИИ в нормальные объекты Task
    from src.task_parser import Task, Priority
    clean_tasks = []
    raw_tasks = analysis.get('tasks', [])
    
    for i, t in enumerate(raw_tasks):
        # Гибкая проверка приоритета (чтобы не ломалось)
        p_raw = str(t.get('priority', 'MEDIUM')).upper()
        if 'HIGH' in p_raw or 'ВЫСОК' in p_raw or 'СРОЧН' in p_raw:
            p_final = Priority.HIGH
        elif 'LOW' in p_raw or 'НИЗК' in p_raw or 'ПОЗЖ' in p_raw:
            p_final = Priority.LOW
        else:
            p_final = Priority.MEDIUM

        clean_tasks.append(Task(
            id=i + 1,
            text=t.get('text', 'Без названия'),
            priority=p_final,
            category=t.get('category', 'другое')
        ))

    print("\n[3/3] Форматирование и сохранение...")
    
    # ВЫВОД КРАСИВОГО СПИСОКА (используем наш обновленный парсер)
    print(pipeline['parser'].format_for_output(clean_tasks))
    
    # СОХРАНЕНИЕ В ФАЙЛ (теперь без ошибок)
    json_data = pipeline['output'].format_as_json(clean_tasks, transcription['text'])
    json_file = pipeline['output'].save_to_file(json_data)
    
    print(f"✅ Готово! Список сохранен в: {json_file}")
    
    return clean_tasks


def main():
    print("=" * 60)
    print("VoiceToTaskApp - Zapusk...")
    print("=" * 60)

    # Zagruzhaem modeli (eto mozhet zanyat vremya)
    pipeline = create_pipeline()
    
    print("\n>> Sistema gotova k rabote!")
    
    while True:
        print("\n" + "-" * 30)
        print("VYBERITE REZHIM:")
        print("1. Zapisyvat golos (mikrofon)")
        print("2. Zagruzit gotovyy file (wav/mp3)")
        print("Q. Vyyti iz programmy")
        
        choice = input("\nVash vybor: ").strip().lower()

        if not choice:
            continue

        if choice == '1':
            print("\n>> Podgotovka mikrofona...")
            try:
                # Zapisyvaem
                audio_path = pipeline['audio_input'].record_mic()
                # Srazu obrabatyvaem
                process_audio_file(str(audio_path), pipeline)
            except Exception as e:
                print(f"ERROR: Oshibka pri zapisi: {e}")
                
        elif choice == '2':
            path_input = input("Vvedite put k faylu: ").strip(' "')
            if os.path.exists(path_input):
                process_audio_file(path_input, pipeline)
            else:
                print("ERROR: File ne najden!")
                
        elif choice == 'q':
            print("Vykhod iz programmy...")
            break
        else:
            print(f"WARNING: Nevernyy vybor: '{choice}'. Vvedite 1, 2 ili Q.")

if __name__ == "__main__":
    main()
