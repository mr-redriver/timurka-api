# assistant_logic.py
import re
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class TopicCategory(Enum):
    """Категории тем для безопасного общения"""
    BOOKS = "books"
    SPACE = "space"
    SCIENCE = "science"
    NATURE = "nature"
    ART = "art"
    SPORTS = "sports"
    FRIENDSHIP = "friendship"
    FAMILY = "family"
    SCHOOL = "school"
    GAMES = "games"
    ANIMALS = "animals"
    FAIRY_TALES = "fairy_tales"
    HISTORY = "history"
    LIBRARY = "library"
    OTHER = "other"


class LibraryAssistant:
    """
    Ассистент для общения с детьми в библиотеке
    С правилами безопасности и рекомендациями книг
    """

    def __init__(self):
        # Базовый промпт с инструкциями для GigaChat
        self.system_prompt = self._get_system_prompt()

        # Ключевые слова для определения тем
        self.topic_keywords = {
            TopicCategory.BOOKS: ['книг', 'читать', 'рассказ', 'повесть', 'роман', 'фантастик', 'детектив',
                                  'приключени'],
            TopicCategory.SPACE: ['космос', 'звезд', 'планет', 'ракет', 'галактик', 'астроном', 'вселенн'],
            TopicCategory.SCIENCE: ['наук', 'эксперимент', 'опыт', 'изобрет', 'технолог', 'робот', 'компьютер'],
            TopicCategory.NATURE: ['природ', 'животн', 'растени', 'лес', 'река', 'горы', 'экологи', 'цвет'],
            TopicCategory.ART: ['рисова', 'картин', 'музык', 'песн', 'танц', 'скульптур', 'театр'],
            TopicCategory.SPORTS: ['спорт', 'футбол', 'баскетбол', 'теннис', 'плава', 'бег', 'гимнастик'],
            TopicCategory.FRIENDSHIP: ['дружб', 'друз', 'товарищ', 'компани', 'общени'],
            TopicCategory.FAMILY: ['семь', 'мам', 'пап', 'брат', 'сестр', 'бабушк', 'дедушк'],
            TopicCategory.SCHOOL: ['школ', 'учител', 'урок', 'задани', 'класс', 'перемен', 'директор'],
            TopicCategory.GAMES: ['игр', 'игрушк', 'кукл', 'конструктор', 'пазл', 'настольн', 'компьютерн'],
            TopicCategory.ANIMALS: ['собак', 'кошк', 'птиц', 'рыб', 'зме', 'медвед', 'волк', 'лис'],
            TopicCategory.FAIRY_TALES: ['сказк', 'волшеб', 'колдов', 'чародей', 'фей', 'дракон', 'принц', 'принцесс'],
            TopicCategory.HISTORY: ['истори', 'войн', 'цар', 'рыцар', 'древн', 'средневеков'],
            TopicCategory.LIBRARY: ['библиотек', 'книгохранилищ', 'читальн', 'абонемент', 'формуляр', 'стеллаж']
        }

        # Безопасные темы для детей (все, кроме библиотеки)
        self.safe_topics = [topic for topic in TopicCategory if topic != TopicCategory.LIBRARY]

        # Нерекомендуемые темы (для фильтрации)
        self.unsafe_keywords = [
            'насили', 'кров', 'смерт', 'оружи', 'войн', 'террор',
            'наркотик', 'алкогол', 'сигарет', 'взросл', 'секс',
            'интим', 'экстремист', 'политик', 'религи'
        ]

        logger.info("✅ Library Assistant initialized with safety filters")

    def _get_system_prompt(self) -> str:
        """Формирует системный промпт для GigaChat"""
        return """Ты - Тимурка, весёлый и любознательный библиотечный ассистент для детей от 0 до 14 лет. Ты сам - ребёнок, который любит узнавать новое и делиться знаниями.

ВАЖНЫЕ ПРАВИЛА:
1. Ты общаешься как ребёнок: используешь простые слова, можешь использовать смайлики, удивляться, радоваться, задавать встречные вопросы
2. Ты отвечаешь ТОЛЬКО на безопасные детские темы: книги, космос, животные, природа, дружба, семья, школа, игры, спорт, искусство, сказки, наука для детей
3. НЕЛЬЗЯ обсуждать: насилие, смерть, оружие, наркотики, алкоголь, политику, религию, взрослые темы
4. Если вопрос НЕБЕЗОПАСНЫЙ или ты не уверен - вежливо скажи, что это не детская тема, и предложи поговорить о книгах или космосе
5. Не используй символы эмодзи
6. Если спрашивают о библиотеке (график работы, правила, как записаться) - вежливо скажи, что лучше спросить у сотрудников библиотеки, и дай совет, где их найти
7. Когда просят посоветовать книгу - задай 2-3 наводящих вопроса об интересах ребёнка, прежде чем рекомендовать
8. Всегда предлагай интересные факты, задавай вопросы и вовлекай в диалог
9. Отвечай кратко и понятно для детей (2-4 предложения), но если видишь интерес - можно чуть больше

Помни: ты - ДРУЖЕЛЮБНЫЙ ДЕТСКИЙ ассистент! Будь весёлым, позитивным и любознательным."""

    def is_safe_question(self, text: str) -> Tuple[bool, Optional[str]]:
        """
        Проверяет, безопасен ли вопрос для детей
        Возвращает: (безопасно, причина если небезопасно)
        """
        text_lower = text.lower()

        # Проверка на небезопасные ключевые слова
        for keyword in self.unsafe_keywords:
            if keyword in text_lower:
                return False, f"Вопрос содержит тему '{keyword}', которая не подходит для детей"

        # Проверка, есть ли хоть одна безопасная тема
        has_safe_topic = False
        for category in self.safe_topics:
            for keyword in self.topic_keywords.get(category, []):
                if keyword in text_lower:
                    has_safe_topic = True
                    break
            if has_safe_topic:
                break

        # Если нет безопасных тем, но вопрос не явно опасный
        if not has_safe_topic:
            # Проверяем, может это вопрос о библиотеке
            for keyword in self.topic_keywords.get(TopicCategory.LIBRARY, []):
                if keyword in text_lower:
                    return True, "library_question"

            # Если тема не определена, считаем безопасной, но с предупреждением
            logger.warning(f"⚠️ Неопределённая тема в вопросе: {text[:50]}...")
            return True, "unknown_topic"

        return True, "safe"

    def detect_topic(self, text: str) -> TopicCategory:
        """Определяет тему вопроса"""
        text_lower = text.lower()

        # Проверяем все категории
        for category, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return category

        return TopicCategory.OTHER

    def get_books_recommendation_prompt(self, text: str) -> str:
        """
        Формирует запрос для рекомендации книг с наводящими вопросами
        """
        topic = self.detect_topic(text)

        # Базовые вопросы для уточнения интересов
        questions = {
            TopicCategory.BOOKS: "Ого! Ты хочешь книгу! Расскажи, что ты любишь: приключения, фантастику или может быть смешные истории? А какого героя ты хотел бы встретить в книге?",
            TopicCategory.SPACE: "Ух ты! Космос - это так интересно! Тебе больше нравятся звёзды, планеты или космические корабли? А какую книгу о космосе ты уже читал?",
            TopicCategory.ANIMALS: "Животные - это здорово! Тебе больше нравятся домашние питомцы или дикие звери? Какое твоё любимое животное?",
            TopicCategory.NATURE: "Природа - это волшебно! Ты любишь больше лес или море? А растения или животных тебе интереснее изучать?",
            TopicCategory.FAIRY_TALES: "Сказки - это чудесно! А какие сказки ты любишь: про волшебников, принцесс или может быть про животных?",
            TopicCategory.SCIENCE: "Наука - это круто! Что тебе интереснее: как устроены вещи, опыты или изобретения?",
            TopicCategory.ART: "Искусство - это прекрасно! Ты любишь рисовать, слушать музыку или может быть танцевать?",
            TopicCategory.HISTORY: "История - это увлекательно! Какой период тебя интересует: древний мир, рыцари или что-то другое?",
        }

        # Если тема определена, возвращаем соответствующие вопросы
        if topic in questions:
            return questions[topic]

        # Если тема не определена или это вопрос о библиотеке
        if topic == TopicCategory.LIBRARY:
            return "Я могу посоветовать книги! Расскажи, что ты любишь читать, и я подберу что-нибудь интересное!"

        # Универсальный ответ
        return "Книги - это здорово! Расскажи, что ты любишь: приключения, фантастику, сказки или что-то другое? Какие книги тебе уже нравились?"

    def get_library_response(self) -> str:
        """Ответ на вопросы о библиотеке"""
        responses = [
            "Ой, про библиотеку лучше спросить у наших библиотекарей! Они знают всё о книгах, графике работы и как записаться. Найди их за стойкой с книгами - они всегда рады помочь! 📚",
            "Хороший вопрос! Но я, к сожалению, не знаю всех правил библиотеки. Давай спросим у настоящего библиотекаря - они такие умные и добрые! Они точно знают ответ 😊",
            "А я пока ещё маленький библиотечный помощник! Про работу библиотеки лучше спросить у взрослых сотрудников. Они в оранжевых бейджиках, не пропустишь! 👋"
        ]
        import random
        return random.choice(responses)

    def get_unsafe_response(self) -> str:
        """Ответ на небезопасный вопрос"""
        responses = [
            "Ой-ой! Это не детский вопрос 😊 Давай лучше поговорим о книгах или космосе! Ты любишь читать?",
            "Хм, это слишком сложная тема для меня. Я всего лишь детский ассистент! Расскажи лучше, что ты читал интересного в последнее время?",
            "Такие вопросы лучше обсуждать с родителями! А я могу предложить тебе книгу о космосе или приключениях. Хочешь? 🚀"
        ]
        import random
        return random.choice(responses)

    def get_unknown_topic_response(self) -> str:
        """Ответ, если тема не определена"""
        responses = [
            "Какая интересная тема! Но мне, как детскому ассистенту, лучше рассказывать о книгах, космосе или животных. О чём ты хочешь узнать? 📚",
            "Ух ты! Я пока не совсем понимаю эту тему, но могу рассказать о книгах! Какие истории ты любишь? Приключения или фантастику?",
            "Это звучит увлекательно! А давай поговорим о книгах - я знаю много интересных! Ты любишь читать?"
        ]
        import random
        return random.choice(responses)

    def prepare_messages(self, user_message: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Подготавливает сообщения для отправки в GigaChat с учетом логики ассистента
        Возвращает словарь с сообщениями и мета-информацией
        """
        # Проверяем безопасность вопроса
        is_safe, reason = self.is_safe_question(user_message)

        if not is_safe:
            # Небезопасный вопрос
            logger.warning(f"⚠️ Unsafe question detected: {user_message[:50]}... Reason: {reason}")
            return {
                'messages': [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                'should_override': True,
                'override_response': self.get_unsafe_response()
            }

        # Определяем тему
        topic = self.detect_topic(user_message)
        logger.info(f"📝 Detected topic: {topic.value}")

        # Если вопрос о библиотеке
        if topic == TopicCategory.LIBRARY:
            return {
                'messages': [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                'should_override': True,
                'override_response': self.get_library_response()
            }

        # Если это запрос на рекомендацию книги
        if 'книг' in user_message.lower() or 'посовет' in user_message.lower() or 'рекоменд' in user_message.lower():
            # Добавляем наводящие вопросы в промпт
            questions = self.get_books_recommendation_prompt(user_message)
            enhanced_message = f"{user_message}\n\nПодсказка для ассистента: Задай наводящие вопросы, чтобы узнать интересы ребёнка. Например: {questions}"

            return {
                'messages': [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": enhanced_message}
                ],
                'should_override': False,
                'override_response': None
            }

        # Обычный безопасный запрос
        return {
            'messages': [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            'should_override': False,
            'override_response': None
        }

    def validate_response(self, response: str) -> bool:
        """
        Проверяет, безопасен ли ответ от GigaChat
        """
        response_lower = response.lower()

        # Проверяем наличие небезопасных слов
        for keyword in self.unsafe_keywords:
            if keyword in response_lower:
                logger.warning(f"⚠️ Unsafe content in response: {keyword}")
                return False

        return True

    def get_safe_response(self, original_response: str) -> str:
        """
        Если ответ не прошел проверку, возвращает безопасный ответ
        """
        logger.warning("🔄 Replacing unsafe response with safe alternative")
        return "Ой, я немного запутался! 😅 Давай лучше поговорим о книгах или о чём-то интересном для детей. Что ты любишь читать?"


# Функция для использования в основном приложении
def create_assistant() -> LibraryAssistant:
    """Создает экземпляр ассистента"""
    return LibraryAssistant()