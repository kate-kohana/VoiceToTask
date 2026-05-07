import os
import json
import re
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

class TaskAnalyzer:
    def __init__(self, model_name="qwen2.5-3b-instruct-q4_k_m.gguf"):
        self.repo_id = "Qwen/Qwen2.5-3B-Instruct-GGUF"
        self.model_file = model_name
        
        try:
            model_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=self.model_file,
                local_dir="models/qwen",
                local_dir_use_symlinks=False
            )
            self.llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=os.cpu_count() or 4,
                verbose=False
            )
        except Exception as e:
            print(f"ERROR: {e}")
            raise

    def analyze_text(self, text: str) -> dict:
        # Улучшенный промпт: требуем ТОЛЬКО JSON, без лишних слов
        system_msg = (
            "You are a task extractor. Extract tasks into JSON ONLY. "
            "Do not write any conversational text. Use this format: "
            '{"tasks": [{"text": "task description", "priority": "HIGH/MEDIUM/LOW", "category": "work/home/study"}]}'
        )

        prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        prompt += f"<|im_start|>user\nExtract tasks from: {text}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n{" # Подсказываем модели, что надо начать с фигурной скобки

        output = self.llm(
            prompt,
            max_tokens=800,
            temperature=0.1,
            stop=["<|im_end|>", "<|im_start|>"]
        )

        raw_response = output['choices'][0]['text']
        # Если модель не вернула первую скобку (так как мы ее дали в промпте), добавляем ее
        if not raw_response.startswith('{'):
            raw_response = '{' + raw_response
            
        return self._parse_response(raw_response)

    def _parse_response(self, response: str) -> dict:
        """Очищает ответ от любого текста до и после JSON-блока."""
        try:
            # 1. Ищем самый первый { и самый последний }
            match = re.search(r'(\{.*\})', response, re.DOTALL)
            if match:
                json_str = match.group(1)
                # 2. Убираем возможные артефакты разметки Markdown
                json_str = json_str.replace('```json', '').replace('```', '').strip()
                return json.loads(json_str)
            
            return {"tasks": [], "summary": "JSON блок не найден"}
        except Exception as e:
            # Если все равно ошибка - выводим в консоль для отладки
            print(f"DEBUG: Ошибка разбора JSON. Ответ модели был: {response}")
            return {"tasks": [], "summary": f"Ошибка: {e}"}
