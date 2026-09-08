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

# Импорт RAG системы
from library_rag import LibraryRAGSystem

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

# ИНИЦИАЛИЗАЦИЯ RAG СИСТЕМЫ
logger.info("=" * 60)
logger.info("📚 ИНИЦИАЛИЗАЦИЯ RAG СИСТЕМЫ")
logger.info("=" * 60)

# Конфигурация RAG
RAG_MODEL_NAME = os.getenv('RAG_MODEL_NAME', 'all-MiniLM-L6-v2')
RAG_VECTOR_DIM = int(os.getenv('RAG_VECTOR_DIM', '384'))
RAG_CHUNK_SIZE = int(os.getenv('RAG_CHUNK_SIZE', '800'))
RAG_CHUNK_OVERLAP = int(os.getenv('RAG_CHUNK_OVERLAP', '150'))
RAG_STORE_DIR = os.getenv('RAG_STORE_DIR', 'vector_store')
RAG_DATA_DIR = os.getenv('RAG_DATA_DIR', './data')

try:
    rag_system = LibraryRAGSystem(
        model_name=RAG_MODEL_NAME,
        vector_dim=RAG_VECTOR_DIM,
        chunk_size=RAG_CHUNK_SIZE,
        chunk_overlap=RAG_CHUNK_OVERLAP
    )

    # Проверяем наличие сохраненной базы
    index_path = os.path.join(RAG_STORE_DIR, 'library_index.faiss')
    metadata_path = os.path.join(RAG_STORE_DIR, 'library_metadata.pkl')

    if os.path.exists(index_path) and os.path.exists(metadata_path):
        # Загружаем существующую базу
        rag_system.load(store_dir=RAG_STORE_DIR)
        logger.info(f"✅ RAG система загружена из {RAG_STORE_DIR}")
        logger.info(f"   - Векторов: {rag_system.index.ntotal}")
        logger.info(f"   - Чанков: {len(rag_system.metadata)}")
    else:
        # Проверяем наличие данных для обучения
        if os.path.exists(RAG_DATA_DIR):
            txt_files = list(Path(RAG_DATA_DIR).glob('*.txt'))
            if txt_files:
                logger.info(f"📁 Найдено {len(txt_files)} текстовых файлов в {RAG_DATA_DIR}")
                logger.info("🔄 Выполняется обучение RAG системы...")
                rag_system.train(data_dir=RAG_DATA_DIR, save_dir=RAG_STORE_DIR)
                logger.info("✅ RAG система обучена и сохранена")
            else:
                logger.warning(f"⚠️ В {RAG_DATA_DIR} нет текстовых файлов")
                logger.warning("⚠️ RAG система будет работать без данных")
                rag_system = None
        else:
            logger.warning(f"⚠️ Директория {RAG_DATA_DIR} не найдена")
            logger.warning("⚠️ RAG система будет работать без данных")
            rag_system = None

except Exception as e:
    logger.error(f"❌ Ошибка инициализации RAG системы: {e}")
    logger.warning("⚠️ Продолжаем работу без RAG")
    rag_system = None

logger.info("=" * 60)

# Инициализация GigaChat клиента
client_id = os.getenv('GIGACHAT_CLIENT_ID')
client_secret = os.getenv('GIGACHAT_CLIENT_SECRET')

if not client_id or not client_secret:
    logger.error("❌ GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET не заданы в .env")
    raise ValueError("GigaChat credentials not found in .env file")

gigachat_client = GigaChatClient(client_id, client_secret)

# Инициализация ассистента с RAG системой
assistant = create_assistant(rag_system=rag_system)
if rag_system:
    logger.info("✅ Library Assistant создан с RAG интеграцией")
else:
    logger.info("✅ Library Assistant создан без RAG (работает в базовом режиме)")

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

        if prepared.get('rag_used', False):
            if prepared.get('rag_found', False):
                logger.info("🔍 RAG: информация найдена")
            else:
                logger.info("🔍 RAG: информация не найдена, перенаправление к библиотекарю")

        if prepared.get('should_override', False):
            response_text = prepared['override_response']
            logger.info(f"💬 Using override response: {response_text[:50]}...")
        else:
            messages = prepared['messages']
            response_text = gigachat_client.chat(messages, temperature=0.8)

            if not response_text:
                return jsonify({'error': 'Не удалось получить ответ от ассистента'}), 500

            # Фильтрация ответа для RAG-запросов
            if prepared.get('filter_response', False):
                original_response = response_text
                response_text = assistant.filter_response(response_text, user_message)
                if response_text != original_response:
                    logger.info(f"🔍 Response filtered: {response_text[:50]}...")

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
    health_data = {
        'status': 'ok',
        'message': 'Тимурка готов к работе!',
        'version': '2.0',
        'tts_available': tts is not None,
        'tts_voice': tts.voice_name if tts else None,
        'voices': PiperTTS.get_voice_names() if tts else [],
        'voice_recognition_available': voice_recognizer_manager is not None and voice_recognizer_manager.model_available,
        'rag_available': rag_system is not None,
        'rag_vectors': rag_system.index.ntotal if rag_system and hasattr(rag_system,
                                                                         'index') and rag_system.index else 0,
        'rag_chunks': len(rag_system.metadata) if rag_system and hasattr(rag_system, 'metadata') else 0
    }

    if rag_system:
        logger.info(f"📊 RAG статус: {health_data['rag_vectors']} векторов, {health_data['rag_chunks']} чанков")

    return jsonify(health_data)


@app.route('/api/rag/status', methods=['GET'])
def rag_status():
    """Статус RAG системы"""
    if rag_system is None:
        return jsonify({
            'available': False,
            'message': 'RAG система не инициализирована'
        })

    try:
        return jsonify({
            'available': True,
            'vectors': rag_system.index.ntotal if rag_system.index else 0,
            'chunks': len(rag_system.metadata),
            'model': rag_system.model_name,
            'chunk_size': rag_system.chunk_size,
            'store_dir': RAG_STORE_DIR,
            'data_dir': RAG_DATA_DIR
        })
    except Exception as e:
        return jsonify({
            'available': False,
            'error': str(e)
        }), 500


@app.route('/api/rag/reload', methods=['POST'])
def reload_rag():
    """Перезагрузка RAG системы"""
    global rag_system, assistant

    try:
        logger.info("🔄 Перезагрузка RAG системы...")

        # Создаем новую RAG систему
        new_rag = LibraryRAGSystem(
            model_name=RAG_MODEL_NAME,
            vector_dim=RAG_VECTOR_DIM,
            chunk_size=RAG_CHUNK_SIZE,
            chunk_overlap=RAG_CHUNK_OVERLAP
        )

        # Загружаем базу
        new_rag.load(store_dir=RAG_STORE_DIR)

        # Обновляем ассистента
        rag_system = new_rag
        assistant.set_rag_system(rag_system)
        assistant.clear_rag_cache()

        logger.info(f"✅ RAG система перезагружена: {rag_system.index.ntotal} векторов")

        return jsonify({
            'success': True,
            'vectors': rag_system.index.ntotal,
            'chunks': len(rag_system.metadata),
            'message': 'RAG система успешно перезагружена'
        })
    except Exception as e:
        logger.error(f"❌ Ошибка перезагрузки RAG: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/rag/search', methods=['POST'])
def rag_search():
    """Тестовый поиск в RAG системе (для отладки)"""
    if rag_system is None:
        return jsonify({'error': 'RAG система не доступна'}), 503

    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = data.get('top_k', 3)

        if not query:
            return jsonify({'error': 'Пустой запрос'}), 400

        results = rag_system.search(query, top_k=top_k)

        return jsonify({
            'query': query,
            'results': [
                {
                    'score': r['similarity_score'],
                    'content': r['content'][:500] + '...' if len(r['content']) > 500 else r['content'],
                    'filename': r['metadata']['filename'],
                    'type': r['metadata']['content_type'],
                    'topics': r['metadata']['topics']
                }
                for r in results
            ]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("🚀 Запуск сервера Тимурка v2.0 с RAG, TTS и голосовым вводом...")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=True)