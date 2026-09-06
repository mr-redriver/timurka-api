# app.py
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import logging
import os
import base64
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from gigachat_client import GigaChatClient
from assistant_logic import LibraryAssistant, create_assistant
from speech import PiperTTS
from voice_recognizer import create_voice_recognizer
import urllib3

# Отключаем SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__,
            static_folder='static',
            static_url_path='/static')
CORS(app)

# Инициализация GigaChat клиента
client_id = os.getenv('GIGACHAT_CLIENT_ID')
client_secret = os.getenv('GIGACHAT_CLIENT_SECRET')

if not client_id or not client_secret:
    logger.error("❌ GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET не заданы в .env")
    raise ValueError("GigaChat credentials not found in .env file")

gigachat_client = GigaChatClient(client_id, client_secret)

# Инициализация ассистента
assistant = create_assistant()
logger.info("✅ Library Assistant created")

# Инициализация TTS
try:
    voice_name = os.getenv('TTS_VOICE', 'denis')
    tts = PiperTTS(voice_name=voice_name)
    logger.info(f"✅ TTS initialized with voice: {voice_name}")
except Exception as e:
    logger.warning(f"⚠️ TTS initialization failed: {e}")
    tts = None

# Инициализация распознавателя речи
model_path = os.getenv('VOSK_MODEL_PATH', 'recognize_models/vosk-model-small-ru-0.22')
try:
    voice_recognizer_manager = create_voice_recognizer(model_path)
    if voice_recognizer_manager.model_available:
        logger.info(f"✅ Voice Recognizer initialized with model: {model_path}")
    else:
        logger.warning("⚠️ Voice Recognizer: модель не найдена")
        voice_recognizer_manager = None
except Exception as e:
    logger.warning(f"⚠️ Voice Recognizer initialization failed: {e}")
    voice_recognizer_manager = None


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """Обработка сообщений от пользователя"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        need_audio = data.get('need_audio', True)

        if not user_message:
            return jsonify({'error': 'Пустое сообщение'}), 400

        logger.info(f"📨 Получено сообщение: {user_message[:50]}...")

        prepared = assistant.prepare_messages(user_message)

        if prepared.get('should_override', False):
            response_text = prepared['override_response']
            logger.info(f"💬 Using override response: {response_text[:50]}...")
        else:
            messages = prepared['messages']
            response_text = gigachat_client.chat(messages, temperature=0.8)

            if not response_text:
                return jsonify({'error': 'Не удалось получить ответ от ассистента'}), 500

            if not assistant.validate_response(response_text):
                response_text = assistant.get_safe_response(response_text)

        response_data = {
            'response': response_text,
            'has_audio': False
        }

        if need_audio and tts is not None:
            try:
                audio_text = response_text[:500]
                if len(response_text) > 500:
                    audio_text += " И многое другое..."

                audio_bytes, metadata = tts.text_to_audio(audio_text)
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

                response_data['audio'] = audio_base64
                response_data['audio_format'] = 'wav'
                response_data['has_audio'] = True
                response_data['audio_duration'] = metadata.get('duration', 0)

                logger.info(f"🔊 Audio generated: {metadata.get('duration', 0):.1f}s")

            except Exception as e:
                logger.error(f"❌ TTS error: {e}")

        logger.info(f"✅ Ответ отправлен: {response_text[:50]}...")
        return jsonify(response_data)

    except Exception as e:
        logger.error(f"❌ Ошибка при обработке запроса: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice', methods=['GET'])
def get_voices():
    """Возвращает список доступных голосов"""
    try:
        if tts is None:
            return jsonify({'error': 'TTS не доступен'}), 503

        voices = PiperTTS.get_voice_names()
        return jsonify({
            'voices': voices,
            'current': tts.voice_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice/change', methods=['POST'])
def change_voice():
    """Смена голоса"""
    try:
        data = request.get_json()
        voice_name = data.get('voice')

        if not voice_name:
            return jsonify({'error': 'Не указан голос'}), 400

        if tts is None:
            return jsonify({'error': 'TTS не доступен'}), 503

        available_voices = PiperTTS.get_voice_names()
        if voice_name not in available_voices:
            return jsonify({
                'error': f'Голос {voice_name} не найден. Доступны: {", ".join(available_voices)}'
            }), 400

        tts.change_voice(voice_name)

        return jsonify({
            'success': True,
            'voice': voice_name,
            'available_voices': available_voices
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice/start', methods=['POST'])
def start_voice_recognition():
    """Запуск распознавания речи"""
    if voice_recognizer_manager is None:
        return jsonify({'error': 'Распознавание речи не доступно'}), 503

    try:
        if voice_recognizer_manager.start_recognition():
            return jsonify({'success': True, 'message': 'Распознавание запущено'})
        else:
            return jsonify({'error': 'Не удалось запустить распознавание'}), 500
    except Exception as e:
        logger.error(f"❌ Ошибка запуска распознавания: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice/stop', methods=['POST'])
def stop_voice_recognition():
    """Остановка распознавания речи и получение результата"""
    if voice_recognizer_manager is None:
        return jsonify({'error': 'Распознавание речи не доступно'}), 503

    try:
        text = voice_recognizer_manager.stop_recognition()
        return jsonify({
            'success': True,
            'text': text,
            'has_text': bool(text)
        })
    except Exception as e:
        logger.error(f"❌ Ошибка остановки распознавания: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/voice/status', methods=['GET'])
def voice_recognition_status():
    """Получение статуса распознавания"""
    if voice_recognizer_manager is None:
        return jsonify({'available': False})

    return jsonify({
        'available': True,
        'is_active': voice_recognizer_manager.is_active,
        'last_result': voice_recognizer_manager.result_text
    })


@app.route('/api/health', methods=['GET'])
def health():
    """Проверка работоспособности"""
    return jsonify({
        'status': 'ok',
        'message': 'Тимурка готов к работе!',
        'version': '2.0',
        'tts_available': tts is not None,
        'tts_voice': tts.voice_name if tts else None,
        'voices': PiperTTS.get_voice_names() if tts else [],
        'voice_recognition_available': voice_recognizer_manager is not None and voice_recognizer_manager.model_available
    })


if __name__ == '__main__':
    logger.info("🚀 Запуск сервера Тимурка v2.0 с TTS и голосовым вводом...")
    app.run(host='0.0.0.0', port=5000, debug=True)