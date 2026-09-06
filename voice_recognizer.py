"""
Модуль для распознавания русской речи с использованием Vosk
"""
import json
import wave
import io
import threading
import queue
import time
from typing import Optional, Callable
from vosk import Model, KaldiRecognizer
import pyaudio
import logging

logger = logging.getLogger(__name__)


class VoiceRecognizer:
    """Класс для распознавания речи с микрофона"""

    def __init__(self, model_path: str = "recognize_models/vosk-model-small-ru-0.22",
                 sample_rate: int = 16000,
                 chunk_size: int = 4000):
        """
        Инициализация распознавателя

        Args:
            model_path: Путь к папке с моделью Vosk
            sample_rate: Частота дискретизации (Гц)
            chunk_size: Размер чанка для обработки
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Загрузка модели
        try:
            self.model = Model(model_path)
            logger.info(f"✅ Модель Vosk загружена из {model_path}")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели Vosk: {e}")
            raise

        # Состояние распознавания
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recognizer = None
        self.audio_thread = None
        self.processing_thread = None
        self.pyaudio_instance = None
        self.stream = None

        # Callback для результатов
        self.on_result_callback: Optional[Callable[[str], None]] = None
        self.on_partial_callback: Optional[Callable[[str], None]] = None
        self.on_error_callback: Optional[Callable[[str], None]] = None

        # Текст результата
        self.final_text = ""
        self.partial_text = ""

        logger.info("✅ VoiceRecognizer инициализирован")

    def set_callbacks(self, on_result: Optional[Callable] = None,
                      on_partial: Optional[Callable] = None,
                      on_error: Optional[Callable] = None):
        """
        Установка callback функций

        Args:
            on_result: Вызывается при финальном распознавании (принимает текст)
            on_partial: Вызывается при промежуточном распознавании (принимает текст)
            on_error: Вызывается при ошибке (принимает текст ошибки)
        """
        self.on_result_callback = on_result
        self.on_partial_callback = on_partial
        self.on_error_callback = on_error

    def start_recording(self):
        """Начать запись и распознавание"""
        if self.is_recording:
            logger.warning("Распознавание уже запущено")
            return False

        try:
            # Инициализация PyAudio
            self.pyaudio_instance = pyaudio.PyAudio()

            # Открываем поток для записи
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )

            # Создаем распознаватель
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

            self.is_recording = True
            self.final_text = ""
            self.partial_text = ""

            # Запускаем поток для чтения аудио
            self.audio_thread = threading.Thread(target=self._audio_reader_thread)
            self.audio_thread.daemon = True
            self.audio_thread.start()

            # Запускаем поток для обработки аудио
            self.processing_thread = threading.Thread(target=self._audio_processor_thread)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            logger.info("🎤 Запись и распознавание запущены")
            return True

        except Exception as e:
            error_msg = f"Ошибка запуска распознавания: {e}"
            logger.error(f"❌ {error_msg}")
            if self.on_error_callback:
                self.on_error_callback(error_msg)
            self._cleanup()
            return False

    def stop_recording(self, wait_for_result: bool = True, timeout: float = 3.0) -> str:
        """
        Остановить запись и получить результат

        Args:
            wait_for_result: Ожидать ли завершения распознавания
            timeout: Таймаут ожидания в секундах

        Returns:
            Распознанный текст
        """
        if not self.is_recording:
            logger.warning("Распознавание не запущено")
            return self.final_text

        logger.info("⏹ Остановка записи...")
        self.is_recording = False

        # Останавливаем поток записи
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Ошибка закрытия потока: {e}")

        # Ждем завершения обработки
        if wait_for_result and self.processing_thread:
            start_time = time.time()
            while (self.processing_thread.is_alive() and
                   time.time() - start_time < timeout):
                time.sleep(0.1)

        # Получаем финальный результат
        if self.recognizer:
            try:
                # Получаем финальный результат из распознавателя
                final_result = json.loads(self.recognizer.FinalResult())
                if final_result and 'text' in final_result:
                    text = final_result['text'].strip()
                    if text:
                        self.final_text = text
                        if self.on_result_callback:
                            self.on_result_callback(text)
            except Exception as e:
                logger.error(f"Ошибка получения финального результата: {e}")

        self._cleanup()
        logger.info(f"✅ Распознавание остановлено. Результат: '{self.final_text}'")
        return self.final_text

    def _audio_reader_thread(self):
        """Поток для чтения аудио с микрофона"""
        logger.debug("Аудио-ридер поток запущен")

        try:
            while self.is_recording and self.stream:
                try:
                    # Читаем данные из микрофона
                    data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    # Добавляем в очередь для обработки
                    self.audio_queue.put(data)
                except Exception as e:
                    logger.error(f"Ошибка чтения аудио: {e}")
                    if self.is_recording:
                        if self.on_error_callback:
                            self.on_error_callback(f"Ошибка чтения аудио: {e}")
                    break

        except Exception as e:
            logger.error(f"Критическая ошибка в аудио-ридере: {e}")
        finally:
            logger.debug("Аудио-ридер поток завершен")

    def _audio_processor_thread(self):
        """Поток для обработки аудио и распознавания"""
        logger.debug("Аудио-процессор поток запущен")

        # Буфер для накопления аудио
        audio_buffer = bytearray()

        try:
            while self.is_recording or not self.audio_queue.empty():
                try:
                    # Получаем данные из очереди с таймаутом
                    data = self.audio_queue.get(timeout=0.5)
                    audio_buffer.extend(data)

                    # Обрабатываем накопленные данные (примерно 1-2 секунды)
                    if len(audio_buffer) >= 16000 * 2:  # 1 секунда аудио
                        self._process_audio_chunk(audio_buffer)
                        audio_buffer = bytearray()

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Ошибка обработки аудио: {e}")
                    if self.is_recording:
                        if self.on_error_callback:
                            self.on_error_callback(f"Ошибка обработки аудио: {e}")

            # Обрабатываем остаток аудио
            if len(audio_buffer) > 0:
                self._process_audio_chunk(audio_buffer)

        except Exception as e:
            logger.error(f"Критическая ошибка в аудио-процессоре: {e}")
        finally:
            logger.debug("Аудио-процессор поток завершен")

    def _process_audio_chunk(self, audio_data: bytearray):
        """Обработка чанка аудиоданных"""
        if not self.recognizer:
            return

        try:
            # Создаем WAV файл в памяти
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)

            wav_buffer.seek(0)
            audio_bytes = wav_buffer.read()

            # Отправляем в распознаватель
            if self.recognizer.AcceptWaveform(audio_bytes):
                # Финальный результат
                result = json.loads(self.recognizer.Result())
                text = result.get('text', '').strip()
                if text:
                    self.final_text = text
                    if self.on_result_callback:
                        self.on_result_callback(text)
            else:
                # Промежуточный результат
                partial = json.loads(self.recognizer.PartialResult())
                partial_text = partial.get('partial', '').strip()
                if partial_text and partial_text != self.partial_text:
                    self.partial_text = partial_text
                    if self.on_partial_callback:
                        self.on_partial_callback(partial_text)

        except Exception as e:
            logger.error(f"Ошибка обработки аудио-чанка: {e}")

    def _cleanup(self):
        """Очистка ресурсов"""
        # Останавливаем потоки
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        # Закрываем поток PyAudio
        if self.stream:
            try:
                if not self.stream.is_stopped():
                    self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Ошибка закрытия потока: {e}")
            self.stream = None

        # Завершаем PyAudio
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except Exception as e:
                logger.error(f"Ошибка завершения PyAudio: {e}")
            self.pyaudio_instance = None

        self.recognizer = None

        # Очищаем очередь
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break

    def get_result(self) -> str:
        """Получить последний распознанный текст"""
        return self.final_text or self.partial_text


# Класс для интеграции с Flask приложением
class VoiceRecognizerManager:
    """Менеджер для управления VoiceRecognizer в веб-приложении"""

    def __init__(self, model_path: str = "vosk-model-small-ru-0.22"):
        self.recognizer = None
        self.model_path = model_path
        self.is_active = False
        self.result_text = ""
        self.lock = threading.Lock()

        # Проверяем доступность модели
        try:
            test_model = Model(model_path)
            del test_model
            self.model_available = True
            logger.info("✅ Модель Vosk доступна")
        except Exception as e:
            self.model_available = False
            logger.warning(f"⚠️ Модель Vosk не доступна: {e}")

    def start_recognition(self) -> bool:
        """Запустить распознавание"""
        if not self.model_available:
            logger.error("❌ Модель Vosk не доступна")
            return False

        with self.lock:
            if self.is_active:
                return True

            try:
                self.recognizer = VoiceRecognizer(self.model_path)
                self.recognizer.set_callbacks(
                    on_result=self._on_result,
                    on_partial=self._on_partial,
                    on_error=self._on_error
                )

                if self.recognizer.start_recording():
                    self.is_active = True
                    self.result_text = ""
                    return True
                else:
                    self.recognizer = None
                    return False

            except Exception as e:
                logger.error(f"❌ Ошибка запуска распознавания: {e}")
                self.recognizer = None
                self.is_active = False
                return False

    def stop_recognition(self) -> str:
        """Остановить распознавание и вернуть результат"""
        with self.lock:
            self.is_active = False

            if not self.recognizer:
                return self.result_text

            try:
                text = self.recognizer.stop_recording(wait_for_result=True)
                if text:
                    self.result_text = text
                return text
            except Exception as e:
                logger.error(f"❌ Ошибка остановки распознавания: {e}")
                return self.result_text
            finally:
                self.recognizer = None

    def _on_result(self, text: str):
        """Callback для финального результата"""
        logger.info(f"📝 Финальный результат: {text}")
        self.result_text = text

    def _on_partial(self, text: str):
        """Callback для промежуточного результата"""
        logger.debug(f"⏳ Частичный результат: {text}")

    def _on_error(self, error: str):
        """Callback для ошибок"""
        logger.error(f"❌ Ошибка распознавания: {error}")


# Фабрика для создания менеджера
def create_voice_recognizer(model_path: str = "vosk-model-small-ru-0.22") -> VoiceRecognizerManager:
    """Создать менеджер распознавания речи"""
    return VoiceRecognizerManager(model_path)