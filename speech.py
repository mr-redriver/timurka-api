#!/usr/bin/env python3
"""
Piper TTS модуль для озвучивания текста
Поддерживает множество русских моделей (голосов)
Модели хранятся в папке voices/ относительно скрипта
"""

import os
import sys
import wave
import re
import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

# Определяем пути относительно текущего файла
CURRENT_DIR = Path(__file__).parent.absolute()
VOICES_DIR = CURRENT_DIR / "voices"


@dataclass
class VoiceModel:
    """Информация о голосовой модели"""
    name: str  # Название модели (например "denis")
    full_name: str  # Полное имя файла (например "ru_RU-denis-medium")
    path: Path  # Путь к ONNX файлу
    json_path: Path  # Путь к JSON файлу
    language: str = "ru_RU"
    quality: str = "medium"

    def __str__(self):
        return f"{self.name} ({self.language}, {self.quality})"


class PiperTTS:
    """Класс для работы с Piper TTS с поддержкой разных голосов"""

    # Доступные голоса (автоматически сканируются из папки voices)
    _available_voices: Dict[str, VoiceModel] = {}

    def __init__(self, voice_name: str = "denis", auto_download: bool = False):
        """
        Инициализация TTS с выбором голоса

        Args:
            voice_name: название голоса (denis, dmitri, irina, ruslan)
            auto_download: автоматически скачать модель, если отсутствует (отключено по умолчанию)
        """
        self.voice_name = voice_name
        self.voice_model = None
        self.voice = None
        self.auto_download = auto_download

        # Сканируем доступные голоса
        self._scan_available_voices()

        # Загружаем выбранный голос
        self._load_voice(voice_name)

    @classmethod
    def _scan_available_voices(cls):
        """Сканирует папку voices и находит все доступные модели"""
        if cls._available_voices:
            return cls._available_voices

        if not VOICES_DIR.exists():
            return {}

        # Ищем все ONNX файлы в структуре ru_RU/*/medium/
        for onnx_file in VOICES_DIR.glob("ru_RU/*/medium/*.onnx"):
            # Парсим имя файла: ru_RU-denis-medium.onnx
            filename = onnx_file.stem  # ru_RU-denis-medium
            parts = filename.split('-')

            if len(parts) >= 3:
                language = parts[0]  # ru_RU
                voice = parts[1]  # denis, dmitri, irina, ruslan
                quality = parts[2]  # medium

                json_file = onnx_file.with_suffix(".onnx.json")

                voice_model = VoiceModel(
                    name=voice,
                    full_name=filename,
                    path=onnx_file,
                    json_path=json_file,
                    language=language,
                    quality=quality
                )
                cls._available_voices[voice] = voice_model
                cls._available_voices[filename] = voice_model  # Также по полному имени

        return cls._available_voices

    @classmethod
    def list_voices(cls) -> List[VoiceModel]:
        """Возвращает список доступных голосов"""
        cls._scan_available_voices()
        return list(cls._available_voices.values())

    @classmethod
    def get_voice_names(cls) -> List[str]:
        """Возвращает список имен доступных голосов"""
        cls._scan_available_voices()
        return list(set([v.name for v in cls._available_voices.values()]))

    def _load_voice(self, voice_name: str):
        """Загружает указанный голос"""
        try:
            from piper import PiperVoice
        except ImportError:
            raise ImportError("Установите piper-tts: pip install piper-tts")

        # Ищем модель
        available_voices = self._scan_available_voices()

        # Поиск по имени или полному имени
        if voice_name in available_voices:
            self.voice_model = available_voices[voice_name]
        elif f"ru_RU-{voice_name}-medium" in available_voices:
            self.voice_model = available_voices[f"ru_RU-{voice_name}-medium"]
        else:
            # Выводим доступные голоса
            available_names = self.get_voice_names()
            raise ValueError(
                f"Голос '{voice_name}' не найден. "
                f"Доступные голоса: {', '.join(available_names)}"
            )

        # Проверяем наличие файлов
        if not self.voice_model.path.exists():
            if self.auto_download:
                self.download_model()
            else:
                raise FileNotFoundError(
                    f"Модель не найдена: {self.voice_model.path}\n"
                    f"Поместите файлы моделей в папку: {VOICES_DIR}/ru_RU/{self.voice_model.name}/medium/"
                )

        # Загружаем модель
        print(f"🎤 Загрузка голоса: {self.voice_model}")
        self.voice = PiperVoice.load(str(self.voice_model.path))
        print(f"✅ Голос загружен (частота: {self.voice.config.sample_rate} Гц)")

    def download_model(self):
        """Скачивает модель из Hugging Face (если включено)"""
        if not self.auto_download:
            raise RuntimeError("Автоматическое скачивание отключено")

        try:
            import requests
        except ImportError:
            raise ImportError("Установите requests: pip install requests")

        # Создаем директорию для модели
        self.voice_model.path.parent.mkdir(parents=True, exist_ok=True)

        model_url = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{self.voice_model.language}/{self.voice_model.language}/{self.voice_model.name}/{self.voice_model.quality}/{self.voice_model.full_name}.onnx"
        json_url = model_url + ".json"

        print(f"\n📥 Скачивание модели {self.voice_model.full_name}...")
        try:
            # Скачиваем ONNX файл
            response = requests.get(model_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            with open(self.voice_model.path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Прогресс: {percent:.1f}%", end='', flush=True)
            print(f"\n✅ Модель сохранена в {self.voice_model.path}")

            # Скачиваем JSON конфиг
            print(f"\n📥 Скачивание {self.voice_model.json_path.name}...")
            response = requests.get(json_url)
            response.raise_for_status()
            with open(self.voice_model.json_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ JSON сохранён в {self.voice_model.json_path}")

        except Exception as e:
            raise Exception(f"Ошибка скачивания модели: {e}")

    def change_voice(self, voice_name: str):
        """Сменить голос без пересоздания объекта"""
        self._load_voice(voice_name)
        self.voice_name = voice_name


    def text_to_audio(self, text: str, output_file: Optional[Path] = None) -> Tuple[bytes, Dict[str, Any]]:
        """
        Преобразует текст в аудио и возвращает WAV данные в bytes

        Args:
            text: текст для озвучивания
            output_file: опциональный путь для сохранения файла

        Returns:
            tuple: (audio_bytes, metadata) где metadata содержит:
                - sample_rate: частота дискретизации
                - duration: длительность в секундах
                - num_frames: количество кадров
                - voice: использованный голос
        """
        if not self.voice:
            raise RuntimeError("Модель не загружена")

        if not text or not text.strip():
            raise ValueError("Текст не может быть пустым")

        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Смайлики
            "\U0001F300-\U0001F5FF"  # Символы и пиктограммы
            "\U0001F680-\U0001F6FF"  # Транспорт и карты
            "\U0001F700-\U0001F77F"  # Алхимические символы
            "\U0001F780-\U0001F7FF"  # Геометрические фигуры
            "\U0001F800-\U0001F8FF"  # Дополнительные стрелки
            "\U0001F900-\U0001F9FF"  # Дополнительные символы и эмодзи
            "\U0001FA00-\U0001FA6F"  # Дополнительные символы
            "\U0001FA70-\U0001FAFF"  # Дополнительные символы
            "\U00002702-\U000027B0"  # Символы
            "\U000024C2-\U0001F251"  # Символы в кружках
            "]+",
            flags=re.UNICODE
        )

        text = emoji_pattern.sub('', text)

        # Если указан output_file, используем его, иначе создаем временный файл
        if output_file is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = Path(temp_file.name)
            temp_file.close()
        else:
            output_path = Path(output_file)

        try:
            # Синтезируем речь напрямую в WAV файл
            with wave.open(str(output_path), 'wb') as wav_file:
                self.voice.synthesize_wav(text, wav_file)

            # Читаем аудио данные в bytes
            with open(output_path, 'rb') as f:
                audio_bytes = f.read()

            # Получаем метаданные
            with wave.open(str(output_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / rate

            metadata = {
                'sample_rate': rate,
                'duration': duration,
                'num_frames': frames,
                'text': text,
                'voice': self.voice_name,
                'model_path': str(self.voice_model.path)
            }

            # Если файл временный, удаляем его
            if output_file is None:
                output_path.unlink()

            return audio_bytes, metadata

        except Exception as e:
            # Очищаем временный файл в случае ошибки
            if output_file is None and output_path.exists():
                output_path.unlink()
            raise Exception(f"Ошибка синтеза речи: {e}")

    def text_to_audio_file(self, text: str, output_file: Path) -> Dict[str, Any]:
        """
        Преобразует текст в аудио и сохраняет в файл

        Args:
            text: текст для озвучивания
            output_file: путь для сохранения WAV файла

        Returns:
            dict: метаданные аудио (частота, длительность и т.д.)
        """
        audio_bytes, metadata = self.text_to_audio(text, output_file)

        # Убеждаемся, что файл существует (если был создан временный, копируем)
        if not Path(output_file).exists():
            with open(output_file, 'wb') as f:
                f.write(audio_bytes)

        return metadata

    def get_info(self) -> Dict[str, Any]:
        """Возвращает информацию о загруженной модели"""
        if not self.voice:
            return {}

        return {
            'voice': self.voice_name,
            'sample_rate': self.voice.config.sample_rate,
            'num_channels': self.voice.config.num_channels,
            'audio_format': self.voice.config.audio_format,
            'model_path': str(self.voice_model.path),
            'model_full_name': self.voice_model.full_name
        }


def text_to_speech(text: str, voice: str = "denis", output_file: Optional[Path] = None) -> Tuple[bytes, Dict[str, Any]]:
    """
    Простая функция для преобразования текста в речь с выбором голоса

    Args:
        text: текст для озвучивания
        voice: название голоса (denis, dmitri, irina, ruslan)
        output_file: опциональный путь для сохранения файла

    Returns:
        tuple: (audio_bytes, metadata)

    Пример:
        # Использование разных голосов
        audio_data, meta = text_to_speech("Привет мир!", voice="denis")
        audio_data, meta = text_to_speech("Привет мир!", voice="irina")
        audio_data, meta = text_to_speech("Привет мир!", voice="dmitri")

        # Сохранить в файл
        text_to_speech("Привет мир!", voice="ruslan", output_file=Path("output.wav"))
    """
    tts = PiperTTS(voice_name=voice)
    return tts.text_to_audio(text, output_file)


# Пример использования
if __name__ == "__main__":
    print("=" * 60)
    print("🔊 Piper TTS - преобразование текста в речь")
    print("=" * 60)

    # Сканируем доступные голоса
    available_voices = PiperTTS.list_voices()
    print(f"\n🎭 Доступные голоса:")
    for voice in available_voices:
        print(f"  • {voice}")

    # Тестовые фразы для разных голосов
    test_phrases = [
        "Привет, мир! Это тест русского синтеза речи.",
        "Сегодня отличная погода для программирования.",
    ]

    # Тестируем каждый доступный голос
    for voice_model in available_voices:
        print(f"\n{'=' * 60}")
        print(f"🎤 Тестирование голоса: {voice_model.name.upper()}")
        print("=" * 60)

        try:
            # Создаем TTS с выбранным голосом
            tts = PiperTTS(voice_name=voice_model.name)

            # Выводим информацию о модели
            info = tts.get_info()
            print(f"  Частота: {info['sample_rate']} Гц")

            # Тестируем фразы
            for i, phrase in enumerate(test_phrases, 1):
                print(f"\n  Фраза {i}: {phrase[:50]}...")

                # Сохраняем в файл
                output_file = CURRENT_DIR / f"test_{voice_model.name}_{i}.wav"
                metadata = tts.text_to_audio_file(phrase, output_file)

                print(f"    ✅ Сохранено: {output_file.name}")
                print(f"    📊 Длительность: {metadata['duration']:.1f} сек")
                print(f"    📦 Размер: {output_file.stat().st_size / 1024:.1f} КБ")

        except Exception as e:
            print(f"  ❌ Ошибка с голосом {voice_model.name}: {e}")

    # Интерактивный режим с выбором голоса
    print("\n" + "=" * 60)
    print("💬 Интерактивный режим")
    print("=" * 60)

    # Выбор голоса
    voice_names = PiperTTS.get_voice_names()
    print(f"\nДоступные голоса: {', '.join(voice_names)}")

    selected_voice = input(f"\nВыберите голос (по умолчанию {voice_names[0] if voice_names else 'denis'}): ").strip()
    if not selected_voice or selected_voice not in voice_names:
        selected_voice = voice_names[0] if voice_names else "denis"

    print(f"\nВыбран голос: {selected_voice}")
    print("Введите текст для озвучивания (или 'quit' для выхода):")

    tts = PiperTTS(voice_name=selected_voice)

    while True:
        user_text = input("\n📝 Текст: ").strip()
        if user_text.lower() in ['quit', 'exit', 'q']:
            break

        if user_text:
            try:
                # Синтезируем речь
                audio_data, metadata = tts.text_to_audio(user_text)
                print(f"  ✅ Создано ({metadata['duration']:.1f} сек)")

                # Сохраняем во временный файл
                import tempfile
                import platform

                temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                temp_path = Path(temp_file.name)
                temp_file.close()

                tts.text_to_audio_file(user_text, temp_path)

                # Воспроизводим на Windows
                if platform.system() == "Windows":
                    os.startfile(str(temp_path))
                    print("  🔊 Воспроизведение запущено")

                # Удаляем через 5 секунд
                import threading


                def cleanup():
                    import time
                    time.sleep(5)
                    if temp_path.exists():
                        temp_path.unlink()


                threading.Thread(target=cleanup, daemon=True).start()

            except Exception as e:
                print(f"  ❌ Ошибка: {e}")

    print("\n👋 До свидания!")