from dotenv import load_dotenv
load_dotenv()  # Загружает .env файл

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import json
from datetime import datetime
import random
import os
import logging
from contextlib import asynccontextmanager

# Импорт нашей RAG системы
from library_rag import LibraryRAGSystem
from gigachat_client import GigaChatClient

# ========== Настройка логирования ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print(f"GIGACHAT_CLIENT_ID: {os.getenv('GIGACHAT_CLIENT_ID', 'NOT SET')[:20]}...")
print(f"GIGACHAT_CLIENT_SECRET: {'SET' if os.getenv('GIGACHAT_CLIENT_SECRET') else 'NOT SET'}")

# ========== Конфигурация ==========
class Config:
    """Конфигурация приложения"""
    # GigaChat API
    GIGACHAT_CLIENT_ID = os.getenv('GIGACHAT_CLIENT_ID', '')
    GIGACHAT_CLIENT_SECRET = os.getenv('GIGACHAT_CLIENT_SECRET', '')

    # RAG настройки
    VECTOR_STORE_DIR = os.getenv('VECTOR_STORE_DIR', 'vector_store')
    RAG_MODEL_NAME = os.getenv('RAG_MODEL_NAME', 'all-MiniLM-L6-v2')
    TOP_K_RESULTS = int(os.getenv('TOP_K_RESULTS', '3'))
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.35'))

    # WordPress интеграция
    WP_API_KEY = os.getenv('WP_API_KEY', '')

    # Режим работы (без GigaChat если нет ключей)
    USE_GIGACHAT = bool(GIGACHAT_CLIENT_ID and GIGACHAT_CLIENT_SECRET)


# ========== Pydantic модели (сохранение совместимости) ==========
class ChatRequest(BaseModel):
    message: str
    language: str = "ru"
    session_id: str
    user_context: Optional[Dict] = None


class Book(BaseModel):
    id: int
    title: str
    author: str
    year: int
    pages: int
    cover: str
    available: bool
    age_from: int
    genre: str


# ========== База книг и авторов (сохранение оригинальной) ==========
BOOKS_DB = [
    {
        "id": 1,
        "title": "Гэсэр",
        "author": "Народный эпос",
        "year": 2015,
        "pages": 280,
        "cover": "wp-content/plugins/timurka-ai-assistant/static/images/examples/covers/geser.jpg",
        "available": True,
        "age_from": 7,
        "genre": "epic",
        "description": "Героический эпос бурятского народа о могучем богатыре Гэсэре."
    },
    {
        "id": 2,
        "title": "Маленький принц",
        "author": "Антуан де Сент-Экзюпери",
        "year": 1943,
        "pages": 96,
        "cover": "wp-content/plugins/timurka-ai-assistant/static/images/examples/covers/prince.jpg",
        "available": True,
        "age_from": 6,
        "genre": "fairy_tale",
        "description": "Философская сказка о дружбе, любви и ответственности."
    },
    {
        "id": 3,
        "title": "В тайге над Байкалом",
        "author": "Валентин Распутин",
        "year": 1987,
        "pages": 320,
        "cover": "wp-content/plugins/timurka-ai-assistant/static/images/examples/covers/baikal.jpg",
        "available": False,
        "age_from": 10,
        "genre": "nature",
        "description": "Рассказы о природе и легендах Байкала."
    },
    {
        "id": 4,
        "title": "Абай Гэсэр",
        "author": "Бурятский эпос",
        "year": 2020,
        "pages": 450,
        "cover": "wp-content/plugins/timurka-ai-assistant/static/images/examples/covers/abay_geser.jpg",
        "available": True,
        "age_from": 8,
        "genre": "epic",
        "description": "Полное собрание улигеров о Гэсэре на бурятском языке."
    },
    {
        "id": 5,
        "title": "Приключения Незнайки",
        "author": "Николай Носов",
        "year": 1954,
        "pages": 200,
        "cover": "wp-content/plugins/timurka-ai-assistant/static/images/examples/covers/neznaika.jpg",
        "available": True,
        "age_from": 5,
        "genre": "adventure",
        "description": "Весёлые приключения коротышек из Цветочного города."
    },
    {
        "id": 6,
        "title": "Денискины рассказы",
        "author": "Виктор Драгунский",
        "year": 1959,
        "pages": 180,
        "cover": "wp-content/plugins/timurka-ai-assistant/static/images/examples/covers/deniska.jpg",
        "available": True,
        "age_from": 6,
        "genre": "humor",
        "description": "Смешные и поучительные истории о мальчике Дениске."
    }
]

# База авторов с биографиями
AUTHORS_DB = {
    "антуан де сент-экзюпери": {
        "name": "Антуан де Сент-Экзюпери",
        "biography": "Французский писатель, поэт и профессиональный лётчик. Родился в 1900 году в Лионе. Его самое известное произведение — «Маленький принц», переведённое на сотни языков. Погиб во время разведывательного полёта в 1944 году.",
        "birth_year": 1900,
        "death_year": 1944,
        "main_works": ["Маленький принц", "Ночной полёт", "Планета людей"]
    },
    "валентин распутин": {
        "name": "Валентин Распутин",
        "biography": "Русский советский писатель, представитель «деревенской прозы». Родился в 1937 году в Иркутской области. Много писал о Байкале и сибирской природе. Ушёл из жизни в 2015 году.",
        "birth_year": 1937,
        "death_year": 2015,
        "main_works": ["Прощание с Матёрой", "Живи и помни", "Уроки французского"]
    },
    "николай носов": {
        "name": "Николай Носов",
        "biography": "Советский детский писатель-прозаик, драматург. Родился в 1908 году в Киеве. Наиболее известен как автор произведений о Незнайке. Умер в 1976 году.",
        "birth_year": 1908,
        "death_year": 1976,
        "main_works": ["Приключения Незнайки и его друзей", "Незнайка на Луне", "Витя Малеев в школе и дома"]
    },
    "виктор драгунский": {
        "name": "Виктор Драгунский",
        "biography": "Русский советский писатель, автор популярных «Денискиных рассказов». Родился в 1913 году в Нью-Йорке, вырос в Москве. Умер в 1972 году.",
        "birth_year": 1913,
        "death_year": 1972,
        "main_works": ["Денискины рассказы", "Он упал на траву", "Сегодня и ежедневно"]
    }
}


# ========== RAG Чат-бот (адаптированный под FastAPI) ==========
class LibraryRAGChatBot:
    """Интеграция RAG системы с GigaChat для библиотечного ассистента"""

    def __init__(self):
        self.rag = None
        self.gigachat = None
        self.is_available = False
        self._init_rag()
        self._init_gigachat()

    def _init_rag(self):
        """Инициализация RAG системы"""
        try:
            if os.path.exists(Config.VECTOR_STORE_DIR):
                self.rag = LibraryRAGSystem(
                    model_name=Config.RAG_MODEL_NAME,
                    chunk_size=1200,
                    chunk_overlap=200
                )
                self.rag.load(Config.VECTOR_STORE_DIR)
                logger.info(f"RAG system loaded with {self.rag.index.ntotal} vectors")
            else:
                logger.warning(f"Vector store not found at {Config.VECTOR_STORE_DIR}")
                self.rag = None
        except Exception as e:
            logger.error(f"Failed to load RAG system: {e}")
            self.rag = None

    def _init_gigachat(self):
        """Инициализация GigaChat клиента"""
        client_id = os.getenv('GIGACHAT_CLIENT_ID')
        client_secret = os.getenv('GIGACHAT_CLIENT_SECRET')

        if client_id and client_secret and client_id != '' and client_secret != '':
            try:
                self.gigachat = GigaChatClient(client_id, client_secret)
                # Проверяем получение токена
                test_token = self.gigachat.get_access_token()
                if test_token:
                    self.is_available = True
                    logger.info("✅ GigaChat initialized and ready!")
                else:
                    logger.warning("⚠️ GigaChat token test failed")
            except Exception as e:
                logger.error(f"Failed to initialize GigaChat: {e}")
                self.gigachat = None
                self.is_available = False
        else:
            logger.warning("⚠️ GigaChat credentials not set in .env file")
            self.gigachat = None
            self.is_available = False

    def _build_system_prompt(self, context: str, question: str, language: str) -> str:
        """Построение системного промпта"""
        lang_name = "русском" if language == "ru" else "бурятском"

        return f"""Ты - интеллектуальный библиотечный ассистент по имени Тимурка. Создан в рамках национального проекта «Семья».

Используй следующий КОНТЕКСТ из базы знаний библиотеки для ответа:

{context}

Вопрос пользователя: {question}

ПРАВИЛА ОТВЕТА:
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. Если в контексте нет информации - используй свои знания библиотекаря, но честно укажи, что информация из общих источников
3. Для вопросов о мероприятиях обязательно указывай ДАТУ, ВРЕМЯ и МЕСТО
4. Для вопросов о книгах давай рекомендации с авторами и названиями
5. Отвечай на {lang_name} языке
6. Будь дружелюбным и полезным, как библиотекарь
7. Если спрашивают о погоде - дружелюбно предложи почитать книгу

Ты - Тимурка, библиотечный помощник. Всегда помогай найти интересные книги и информацию о библиотеке!"""

    def _search_books_in_db(self, query: str) -> List[Dict]:
        """Поиск книг в локальной базе"""
        query_lower = query.lower()
        results = []

        for book in BOOKS_DB:
            if (query_lower in book["title"].lower() or
                    query_lower in book["author"].lower() or
                    any(genre in query_lower for genre in [book["genre"]]) or
                    (query_lower.isdigit() and int(query_lower) == book["age_from"])):
                results.append(book)

        return results[:5]

    def _get_rag_context(self, question: str) -> tuple:
        """Получение контекста из RAG системы"""
        if not self.rag:
            return "", 0, None

        try:
            results = self.rag.search(question, top_k=Config.TOP_K_RESULTS,
                                      similarity_threshold=Config.SIMILARITY_THRESHOLD)

            if not results:
                return "", 0, None

            context_parts = []
            avg_score = 0
            for res in results:
                context_parts.append(f"[Источник: {res['metadata']['filename']}]\n{res['content']}")
                avg_score += res['similarity_score']

            avg_score /= len(results)
            context = "\n\n---\n\n".join(context_parts)

            # Ограничиваем длину
            if len(context) > 3000:
                context = context[:3000] + "..."

            return context, avg_score, results

        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return "", 0, None

    async def ask(self, question: str, language: str = "ru") -> Dict:
        """
        Основной метод для получения ответа от бота
        """

        # Определяем тип вопроса
        question_type = self._detect_question_type(question)

        # 1. Сначала пробуем RAG + GigaChat
        if self.is_available and self.rag:
            try:
                context, similarity, rag_results = self._get_rag_context(question)

                logger.info(
                    f"RAG search: similarity={similarity}, results={len(rag_results) if rag_results else 0}, question_type={question_type}")

                if context and similarity > 0.35:
                    system_prompt = self._build_system_prompt(context, question, language)

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ]

                    answer = self.gigachat.chat(messages, temperature=0.7, max_tokens=1000)

                    if answer:
                        # ВАЖНО: КНИГИ ДОБАВЛЯЕМ ТОЛЬКО ДЛЯ ВОПРОСОВ О КНИГАХ!
                        recommended_books = []
                        if question_type == 'books':
                            recommended_books = self._extract_books_from_rag(rag_results)
                            if not recommended_books:
                                recommended_books = self._search_books_in_db(question)

                        # Для мероприятий и правил - НИКОГДА не добавляем книги
                        logger.info(
                            f"✅ RAG response, question_type={question_type}, books_found={len(recommended_books)}")
                        return {
                            "message": answer,
                            "books": recommended_books,  # Пустой массив для не-книжных вопросов
                            "author": None,
                            "suggestions": self._get_suggestions_by_type(question_type, language),
                            "bookNotFound": False,
                            "alternatives": [],
                            "from_rag": True,
                            "similarity_score": similarity
                        }
            except Exception as e:
                logger.error(f"RAG/GigaChat error: {e}")

        # 2. Если RAG не сработал И вопрос о книгах - ищем в локальной базе
        if question_type == 'books':
            books_found = self._search_books_in_db(question)

            if books_found:
                logger.info(f"Found {len(books_found)} books in local DB")
                message = self._format_book_response(books_found, question, language)
                return {
                    "message": message,
                    "books": books_found,
                    "author": None,
                    "suggestions": self._get_suggestions_by_type('books', language),
                    "bookNotFound": False,
                    "alternatives": [],
                    "from_rag": False
                }

        # 3. Если RAG не сработал И вопрос НЕ о книгах - возвращаем ответ БЕЗ КНИГ
        fallback_messages = {
            'events': {
                'ru': "Извините, я не нашёл информации о мероприятиях в библиотеке. Попробуйте спросить о книгах или обратитесь к библиотекарю.",
                'bua': "Уучлаарай, би номой сангай арга хэмжээнүүдэй тухай мэдээлэл оложо чадахгүй байна."
            },
            'rules': {
                'ru': "Извините, я не нашёл информации о правилах библиотеки. Попробуйте спросить о книгах или обратитесь к библиотекарю.",
                'bua': "Уучлаарай, би номой сангай дүрэмнүүдэй тухай мэдээлэл оложо чадахгүй байна."
            },
            'general': {
                'ru': "Извините, я не смог найти ответ на ваш вопрос. Попробуйте уточнить запрос или спросить о конкретной книге.",
                'bua': "Уучлаарай, би танай асуултада хариулта оложо чадахгүй байна."
            }
        }

        fb = fallback_messages.get(question_type, fallback_messages['general'])
        return {
            "message": fb.get(language, fb['ru']),
            "books": [],  # ВАЖНО: пустой массив, никаких книг!
            "author": None,
            "suggestions": self._get_suggestions_by_type(question_type, language),
            "bookNotFound": False,
            "alternatives": [],
            "from_rag": False
        }

    def _detect_question_type(self, question: str) -> str:
        """Определяет тип вопроса с более строгими правилами"""
        question_lower = question.lower()

        # Приоритет 1: Мероприятия (самый высокий приоритет)
        events_keywords = [
            'мероприяти', 'квартирник', 'библионочь', 'лекци', 'встреч',
            'выставк', 'концерт', 'фестивал', 'праздник', 'клуб', 'кружок',
            'мастер-класс', 'заняти', 'турнир', 'собрани', 'событи',
            'анонс', 'расписани', 'когда будет', 'во сколько', 'где проходит'
        ]

        # Приоритет 2: Правила
        rules_keywords = [
            'правил', 'как записаться', 'забронировать', 'бронь', 'выдача',
            'возврат', 'просрочк', 'штраф', 'задолженност', 'продлит',
            'читательский билет', 'график', 'режим работы', 'часы работы',
            'запись в библиотеку', 'электронный ресурс', 'литрес', 'нэб'
        ]

        # Приоритет 3: Книги (только если нет ключевых слов мероприятий и правил)
        books_keywords = [
            'книг', 'сказк', 'рассказ', 'роман', 'повест', 'поэм', 'стих',
            'эпос', 'фантастик', 'детектив', 'приключени', 'энциклопед',
            'читать', 'почитать', 'прочитать', 'автор', 'писател', 'жанр',
            'обложк', 'страниц', 'издательств'
        ]

        # Сначала проверяем мероприятия
        if any(kw in question_lower for kw in events_keywords):
            return 'events'

        # Потом правила
        if any(kw in question_lower for kw in rules_keywords):
            return 'rules'

        # Потом книги
        if any(kw in question_lower for kw in books_keywords):
            return 'books'

        return 'general'

    def _get_suggestions_by_type(self, question_type: str, language: str) -> List[str]:
        """Возвращает подсказки в зависимости от типа вопроса"""
        if language == 'bua':
            if question_type == 'events':
                return [
                    '🎭 Үзэсгэлэнүүд',
                    '📚 Номой хүдэлмэри',
                    '🎨 Мастер-классууд'
                ]
            elif question_type == 'rules':
                return [
                    '📝 Бүртсэлгэ',
                    '⏰ Ажалай саг',
                    '📖 Номой бусаалга'
                ]
            else:
                return [
                    '📚 Хүүгэдтэ номнууд',
                    '🐉 Буряад үлгэрүүд',
                    '⭐ Юу уншахаб?'
                ]
        else:
            if question_type == 'events':
                return [
                    '🎭 Библионочь 2026',
                    '🎨 Мастер-классы',
                    '📚 Книжные выставки'
                ]
            elif question_type == 'rules':
                return [
                    '📝 Как записаться в библиотеку',
                    '⏰ График работы',
                    '📖 Как продлить книгу онлайн'
                ]
            else:
                return [
                    '📚 Книги для 7 лет',
                    '🐉 Бурятские народные сказки',
                    '⭐ Что почитать на ночь?'
                ]

    def _format_book_response(self, books: List[Dict], query: str, language: str) -> str:
        """Форматирование ответа для найденных книг (только для книжных вопросов)"""
        if not books:
            return self._get_no_books_response(language)

        if language == "bua":
            if len(books) == 1:
                return f"Би ном олоб! 📚\n\n**{books[0]['title']}**\n{books[0].get('author', 'Автор тодорхойгүй')}\n\n{books[0].get('description', '')[:200]}"
            else:
                return f"Тоохоной {len(books)} ном олоб! 📚\n\n" + "\n\n".join([
                    f"**{b['title']}** — {b.get('author', 'Автор тодорхойгүй')}\n{b.get('description', '')[:100]}..."
                    for b in books[:3]
                ])
        else:
            if len(books) == 1:
                return f"Нашёл для тебя книгу! 📚\n\n**{books[0]['title']}**\n{books[0].get('author', 'Автор не указан')}\n\n{books[0].get('description', '')[:200]}"
            else:
                return f"Нашёл для тебя {len(books)} книг! 📚\n\n" + "\n\n".join([
                    f"**{b['title']}** — {b.get('author', 'Автор не указан')}\n{b.get('description', '')[:100]}..."
                    for b in books[:3]
                ])

    def _get_no_books_response(self, language: str) -> str:
        """Ответ когда книги не найдены"""
        if language == "bua":
            return "Уучлаарай, би танай хүсэлтөөр ном оложо чадахгүй байна. Өөр нэрэ, автор эсэбэл жанрай оролдоно уу."
        else:
            return "Извините, я не смог найти книги по вашему запросу. Попробуйте другое название, автора или жанр."

    def _get_fallback_response(self, language: str) -> str:
        """Fallback ответ когда ничего не найдено"""
        if language == "bua":
            return "Уучлаарай, би танай асуултада хариулта оложо чадахгүй байна. Та номой нэрэ, автор эсэбэл жанрай тодорхойлно уу."
        else:
            return "Извините, я не смог найти ответ на ваш вопрос. Попробуйте уточнить запрос или спросить о конкретной книге, авторе или жанре."

    def _extract_books_from_rag(self, rag_results: List) -> List[Dict]:
        """Извлечение книг из результатов RAG поиска"""
        # Пытаемся найти упоминания книг в RAG результатах
        # и сопоставить с локальной базой
        extracted = []
        for result in rag_results:
            content = result['content'].lower()
            for book in BOOKS_DB:
                if (book['title'].lower() in content or
                        book['author'].lower() in content):
                    if book not in extracted and book['available']:
                        extracted.append(book)
                        if len(extracted) >= 3:
                            return extracted
        return extracted


# ========== Глобальные объекты ==========
chatbot = None


# ========== Lifespan для FastAPI ==========
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global chatbot
    logger.info("Starting up Library RAG Chatbot...")
    chatbot = LibraryRAGChatBot()
    logger.info(f"Chatbot available: {chatbot.is_available}")
    yield
    # Shutdown
    logger.info("Shutting down...")


# ========== FastAPI приложение ==========
app = FastAPI(
    title="Library RAG Chatbot - Timurka",
    description="Интеллектуальный библиотечный ассистент с RAG системой",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware (сохранение оригинальной конфигурации)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== Оригинальные эндпоинты (сохранение совместимости) ==========

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Основной эндпоинт для чата"""
    try:
        result = await chatbot.ask(request.message, request.language)

        # Убедитесь, что возвращаете правильный формат
        return {
            "message": result["message"],
            "books": result.get("books", []),
            "author": result.get("author"),
            "suggestions": result.get("suggestions", []),
            "bookNotFound": result.get("bookNotFound", False),
            "alternatives": result.get("alternatives", []),
            "from_rag": result.get("from_rag", False),
            "session_id": request.session_id
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {
            "message": "Извините, произошла ошибка. Пожалуйста, попробуйте позже.",
            "books": get_recommendations(3),
            "suggestions": get_suggestions(request.language),
            "error": str(e)
        }

@app.get("/api/greeting")
async def greeting(lang: str = "ru"):
    """Приветственное сообщение"""
    greetings = {
        "ru": "Привет! Я Тимурка 🤖 Создан в рамках национального проекта «Семья». Хочешь найти интересную книгу? Расскажи, что ты любишь читать, и я помогу!",
        "bua": "Сайн байна! Би Тимурка 🤖 Национальный проект «Семья»-да бүтээгдэһэн. Ном хайха дуратай гүш? Юун дуртайгаа хэлээш, би тусалаяб!"
    }
    return {"message": greetings.get(lang, greetings["ru"])}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket эндпоинт для реального времени"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Обработка WebSocket сообщений
            try:
                json_data = json.loads(data)
                question = json_data.get("message", "")
                language = json_data.get("language", "ru")

                result = await chatbot.ask(question, language)
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "data": result
                }))
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")


# ========== Новые эндпоинты для RAG функциональности ==========

@app.get("/api/rag/status")
async def rag_status():
    """Проверка статуса RAG системы"""
    return {
        "rag_available": chatbot.rag is not None,
        "gigachat_available": chatbot.is_available,
        "vector_count": chatbot.rag.index.ntotal if chatbot.rag else 0,
        "top_k": Config.TOP_K_RESULTS,
        "similarity_threshold": Config.SIMILARITY_THRESHOLD
    }


@app.post("/api/rag/search")
async def rag_search(request: ChatRequest):
    """Прямой поиск в RAG системе (без генерации ответа)"""
    if not chatbot.rag:
        return {"error": "RAG system not available"}, 503

    results = chatbot.rag.search(request.message, top_k=5, similarity_threshold=0.3)

    return {
        "query": request.message,
        "results": [
            {
                "content": r["content"][:500] + "..." if len(r["content"]) > 500 else r["content"],
                "similarity": r["similarity_score"],
                "source": r["metadata"]["filename"],
                "type": r["metadata"].get("content_type", "unknown")
            }
            for r in results
        ]
    }


@app.get("/api/health")
async def health_check():
    """Проверка работоспособности"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "rag_loaded": chatbot.rag is not None,
        "gigachat_ready": chatbot.is_available if chatbot else False,
        "version": "2.0.0"
    }


# ========== Вспомогательные функции (сохранение оригинальных) ==========

def detect_intent(message: str) -> str:
    """Определение намерения пользователя"""
    keywords = {
        "find_book": ["найти", "книгу", "поищи", "где", "есть", "расскажи о книге", "почитать", "почитаю", "интересует",
                      "хочу", "рекоменд", "посовет", "что почитать"],
        "author": ["автор", "кто написал", "биография", "писатель", "поэт"],
        "weather": ["погода", "на улице", "холодно", "тепло", "солнце", "дождь"]
    }

    for intent, words in keywords.items():
        if any(word in message for word in words):
            return intent
    return "general"


def get_author_info(query: str, language: str) -> Optional[Dict]:
    """Получение информации об авторе"""
    query_lower = query.lower()

    for author_key in AUTHORS_DB.keys():
        if author_key in query_lower:
            author = AUTHORS_DB[author_key].copy()
            return author

    return None


def get_weather_response(language: str) -> str:
    """Ответ на вопрос о погоде"""
    weather_phrases = {
        "ru": [
            "За окном солнечно! ☀️ Отличная погода, чтобы почитать книжку на свежем воздухе!",
            "На улице небольшой дождик 🌧️ Самое время уютно устроиться дома с интересной книгой!",
            "Сегодня ветрено 🍃 Но в библиотеке всегда тепло и уютно. Приходите за новой книгой!",
            "Погода прекрасная! Идеальный день для чтения в парке 📚"
        ],
        "bua": [
            "Газаа нарантай! ☀️ Ном уншахада тохиромжотой үдэр!",
            "Газаа бороотой 🌧️ Гэртээ ном уншаха дулаахан саг!",
            "Һалхинтай 🍃 Номой санда дулаан. Шэнэ ном абахада ерээрэй!",
            "Погода һайн! Сая ном уншаха үдэр 📚"
        ]
    }

    phrases = weather_phrases.get(language, weather_phrases["ru"])
    return random.choice(phrases)


def generate_response(intent: str, books: List[Dict], author: Optional[Dict], query: str, language: str) -> str:
    """Генерация ответа в зависимости от контекста"""
    responses = {
        "ru": {
            "book_found": f"Нашёл для тебя {len(books)} интересных книг! 📚 Вот что я рекомендую:",
            "book_not_found": "По твоему запросу книг не найдено 🤔 Но я подобрал несколько интересных вариантов:",
            "author": f"📖 Вот что я знаю об этом авторе:",
            "general": f"Отличный вопрос! Давай посмотрим, что у нас есть в библиотеке."
        },
        "bua": {
            "book_found": f"Тоохоной {len(books)} ном олоб! 📚 Эдэные зөвлөжэ байна:",
            "book_not_found": "Тоохоной ном олдогүй 🤔 Гэбэшь би хэдэн сонирхолтой номуудые даалгабарил:",
            "author": f"📖 Энэ зохёолшые тухай юу мэдэхэб:",
            "general": f"Һайн асуулта! Номой санда юу байгаае хараял."
        }
    }

    resp = responses.get(language, responses["ru"])

    if intent == "find_book" and books:
        return resp["book_found"]
    elif intent == "find_book" and not books:
        return resp["book_not_found"]
    elif intent == "author" and author:
        return resp["author"]
    else:
        return resp["general"]


def get_recommendations(limit: int = 3, age: Optional[int] = None) -> List[Dict]:
    """Получение рекомендаций книг"""
    available_books = [book for book in BOOKS_DB if book["available"]]

    if age:
        available_books = [book for book in available_books if book["age_from"] <= age]

    random.shuffle(available_books)
    return available_books[:limit]


def get_suggestions(language: str) -> List[str]:
    """Получение списка подсказок для пользователя"""
    suggestions = {
        "ru": [
            "📚 Книги для первого класса",
            "🎭 Бурятские народные сказки",
            "🦕 Энциклопедии для детей",
            "🐉 Сказки о животных",
            "⭐ Что почитать на ночь?",
            "🏛️ Мероприятия в библиотеке",
            "📖 Электронные ресурсы"
        ],
        "bua": [
            "📚 Нэгэдүгээр класста номнууд",
            "🎭 Буряад арадай үлгэрүүд",
            "🦕 Номой гошоод",
            "🐉 Амитад тухай үлгэрүүд",
            "⭐ Үбшэнөрөө юу уншаха?"
        ]
    }

    return suggestions.get(language, suggestions["ru"])


# ========== Точка входа ==========
if __name__ == "__main__":
    import uvicorn

    # Отключаем предупреждения SSL для тестирования
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # Загрузка переменных окружения из .env
    from dotenv import load_dotenv

    load_dotenv()

    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', '0.0.0.0')
    reload_mode = os.getenv('RELOAD', 'false').lower() == 'true'

    print(f"Starting Library RAG Chatbot on {host}:{port}")
    print(f"GigaChat enabled: {Config.USE_GIGACHAT}")
    print(f"RAG available: {os.path.exists(Config.VECTOR_STORE_DIR)}")

    # Убираем reload=True при запуске скрипта напрямую
    # Или используем правильный синтаксис
    if reload_mode:
        # Режим с авто-перезагрузкой (требует импорта строкой)
        uvicorn.run("main:app", host=host, port=port, reload=True)
    else:
        # Обычный запуск без reload
        uvicorn.run(app, host=host, port=port)