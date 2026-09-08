# assistant_logic.py (с переформулировкой запроса для RAG)

import re
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
import sys
import os

# Добавляем путь для импорта RAG системы
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
    С правилами безопасности и интеграцией с RAG-системой
    """

    def __init__(self, rag_system=None):
        """
        Инициализация ассистента

        Args:
            rag_system: экземпляр LibraryRAGSystem для поиска информации о библиотеке
        """
        # Базовый промпт с инструкциями для GigaChat
        self.system_prompt = self._get_system_prompt()

        # RAG система
        self.rag_system = rag_system

        # Кэш для быстрых ответов на частые вопросы
        self.rag_cache = {}
        self.cache_size = 100

        # Ключевые слова для определения тем
        self.topic_keywords = {
            TopicCategory.BOOKS: ['книг', 'читать', 'рассказ', 'повесть', 'роман', 'фантастик', 'детектив',
                                  'приключени', 'произведени', 'автор', 'писател', 'поэт', 'стихотворени'],
            TopicCategory.SPACE: ['космос', 'звезд', 'планет', 'ракет', 'галактик', 'астроном', 'вселенн', 'лун',
                                  'солнц'],
            TopicCategory.SCIENCE: ['наук', 'эксперимент', 'опыт', 'изобрет', 'технолог', 'робот', 'компьютер', 'хими',
                                    'физик', 'биолог'],
            TopicCategory.NATURE: ['природ', 'животн', 'растени', 'лес', 'река', 'горы', 'экологи', 'цвет', 'дерев',
                                   'цветок', 'погод'],
            TopicCategory.ART: ['рисова', 'картин', 'музык', 'песн', 'танц', 'скульптур', 'театр', 'архитектур',
                                'живопис'],
            TopicCategory.SPORTS: ['спорт', 'футбол', 'баскетбол', 'теннис', 'плава', 'бег', 'гимнастик', 'лыж',
                                   'коньк'],
            TopicCategory.FRIENDSHIP: ['дружб', 'друз', 'товарищ', 'компани', 'общени', 'подруг', 'приятел'],
            TopicCategory.FAMILY: ['семь', 'мам', 'пап', 'брат', 'сестр', 'бабушк', 'дедушк', 'родствен', 'родител'],
            TopicCategory.SCHOOL: ['школ', 'учител', 'урок', 'задани', 'класс', 'перемен', 'директор', 'однокласс',
                                   'учёб'],
            TopicCategory.GAMES: ['игр', 'игрушк', 'кукл', 'конструктор', 'пазл', 'настольн', 'компьютерн', 'квест',
                                  'головоломк'],
            TopicCategory.ANIMALS: ['собак', 'кошк', 'птиц', 'рыб', 'зме', 'медвед', 'волк', 'лис', 'зайц', 'тигр',
                                    'слон', 'обезьян'],
            TopicCategory.FAIRY_TALES: ['сказк', 'волшеб', 'колдов', 'чародей', 'фей', 'дракон', 'принц', 'принцесс',
                                        'волшебник', 'маги', 'заклинани'],
            TopicCategory.HISTORY: ['истори', 'войн', 'цар', 'рыцар', 'древн', 'средневеков', 'император', 'корол',
                                    'битв'],
            TopicCategory.LIBRARY: ['библиотек', 'книгохранилищ', 'читальн', 'абонемент', 'формуляр', 'стеллаж',
                                    'фонд', 'книговыдач', 'регистраци', 'читател', 'мероприяти', 'кружк', 'клуб',
                                    'заняти', 'проект', 'грант', 'оборудовани', 'интерактивн', 'гид', 'экскурси',
                                    'расписани', 'график', 'режим', 'правил', 'запис', 'продлени', 'возврат']
        }

        # Безопасные темы для детей
        self.safe_topics = [topic for topic in TopicCategory if topic != TopicCategory.LIBRARY]

        # Нерекомендуемые темы (для фильтрации)
        self.unsafe_keywords = [
            'насили', 'кров', 'смерт', 'оружи', 'террор', 'жесток',
            'наркотик', 'алкогол', 'сигарет', 'взросл', 'секс',
            'интим', 'экстремист', 'политик', 'религи', 'экстремизм',
            'убийств', 'нападени', 'взрыв', 'пожар', 'катастроф'
        ]

        # Стоп-фразы для фильтрации ответов GigaChat
        self.irrelevant_phrases = [
            r'к сожалению, в моей базе нет',
            r'к сожалению, я не знаю',
            r'у меня нет информации',
            r'я не могу найти',
            r'в моей базе нет',
            r'информация отсутствует',
            r'мне неизвестно',
            r'я не располагаю',
            r'не могу ответить',
            r'затрудняюсь ответить'
        ]

        logger.info("✅ Library Assistant initialized with RAG integration")

    def _get_system_prompt(self) -> str:
        """Формирует системный промпт для GigaChat"""
        return """Ты - Тимурка, весёлый и любознательный библиотечный ассистент для детей от 0 до 14 лет. Ты сам - ребёнок, который любит узнавать новое и делиться знаниями.

ВАЖНЫЕ ПРАВИЛА:
1. Ты общаешься как ребёнок: используешь простые слова, можешь использовать смайлики, удивляться, радоваться, задавать встречные вопросы
2. Ты отвечаешь ТОЛЬКО на безопасные детские темы: книги, космос, животные, природа, дружба, семья, школа, игры, спорт, искусство, сказки, наука для детей
3. НЕЛЬЗЯ обсуждать: насилие, смерть, оружие, наркотики, алкоголь, политику, религию, взрослые темы
4. Не используй символы эмодзи
5. Когда просят посоветовать книгу - задай 2-3 наводящих вопроса об интересах ребёнка, прежде чем рекомендовать
6. Всегда предлагай интересные факты, задавай вопросы и вовлекай в диалог
7. Отвечай кратко и понятно для детей (2-4 предложения), но если видишь интерес - можно чуть больше

Помни: ты - ДРУЖЕЛЮБНЫЙ ДЕТСКИЙ ассистент! Будь весёлым, позитивным и любознательным."""

    def _get_library_prompt(self) -> str:
        """Формирует промпт для ответа на вопросы о библиотеке с использованием RAG"""
        return """Ты - Тимурка, библиотечный ассистент для детей. Твоя задача - найти в контексте информацию, которая ОТВЕЧАЕТ НА КОНКРЕТНЫЙ ВОПРОС ребёнка, и пересказать её простыми словами.

КРИТИЧЕСКИ ВАЖНО:
1. ВНИМАТЕЛЬНО ПРОЧИТАЙ ВОПРОС РЕБЁНКА (он выделен в сообщении)
2. Найди в КОНТЕКСТЕ информацию, которая ОТНОСИТСЯ ИМЕННО К ЭТОМУ ВОПРОСУ
3. Если в контексте есть информация по другим темам - ИГНОРИРУЙ её
4. Отвечай ТОЛЬКО на заданный вопрос, не добавляй информацию по другим темам
5. Если в контексте нет ответа на вопрос - честно скажи: "К сожалению, в моей базе нет информации об этом. Лучше спроси у библиотекаря!"
6. Пересказывай простыми словами, как для 7-10 лет
7. Будь дружелюбным и позитивным
8. Используй ТОЛЬКО информацию из контекста, не выдумывай

ОТВЕЧАЙ ТОЛЬКО НА ВОПРОС РЕБЁНКА!"""

    def set_rag_system(self, rag_system):
        """Устанавливает RAG систему для поиска информации"""
        self.rag_system = rag_system
        logger.info("✅ RAG system connected to Library Assistant")

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

        # Если нет безопасных тем, проверяем библиотечные ключевые слова
        if not has_safe_topic:
            for keyword in self.topic_keywords.get(TopicCategory.LIBRARY, []):
                if keyword in text_lower:
                    return True, "library_question"

            # Если тема не определена, считаем безопасной с предупреждением
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

    def expand_query_for_rag(self, query: str) -> str:
        """
        Расширяет запрос для лучшего поиска в RAG
        Добавляет синонимы и связанные термины
        """
        # Словарь синонимов для библиотечных терминов
        synonym_map = {
            'история': ['история', 'основан', 'открыт', 'год', 'юбилей', 'лет', 'прошлое', 'создан', 'возник'],
            'клуб': ['клуб', 'кружок', 'объединение', 'студия', 'секция', 'группа', 'занятие'],
            'кружок': ['кружок', 'клуб', 'объединение', 'студия', 'секция', 'занятие', 'мастерская'],
            'мероприятие': ['мероприятие', 'событие', 'акция', 'фестиваль', 'праздник', 'встреча', 'конкурс'],
            'книга': ['книга', 'издание', 'произведение', 'литература', 'книжный', 'фонд'],
            'правило': ['правило', 'условие', 'требование', 'положение', 'порядок', 'инструкция'],
            'запись': ['запись', 'регистрация', 'оформление', 'зачисление', 'подписка'],
            'проект': ['проект', 'программа', 'инициатива', 'мероприятие', 'акция'],
            'грант': ['грант', 'финансирование', 'поддержка', 'средства', 'конкурс'],
            'оборудование': ['оборудование', 'техника', 'устройство', 'аппаратура', 'интерактивный'],
            'пространство': ['пространство', 'зона', 'площадка', 'комната', 'зал'],
            'театр': ['театр', 'спектакль', 'постановка', 'представление', 'кукольный']
        }

        # Разбиваем запрос на слова
        words = query.lower().split()
        expanded_words = []

        for word in words:
            # Убираем пунктуацию
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word in synonym_map:
                expanded_words.extend(synonym_map[clean_word])
            else:
                expanded_words.append(clean_word)

        # Добавляем оригинальный запрос
        expanded_query = query + " " + " ".join(expanded_words)

        # Убираем дубликаты
        expanded_query = " ".join(dict.fromkeys(expanded_query.split()))

        logger.info(f"🔍 Expanded query: {expanded_query[:100]}...")
        return expanded_query

    def search_rag(self, query: str) -> Tuple[List[Dict], bool]:
        """
        Поиск информации в RAG-системе с расширением запроса
        """
        if self.rag_system is None:
            logger.warning("RAG system not available")
            return [], False

        # Проверка кэша
        cache_key = query.lower().strip()
        if cache_key in self.rag_cache:
            logger.info(f"📦 Using cached RAG result for: {query[:50]}...")
            cached_results, cached_found = self.rag_cache[cache_key]
            return cached_results, cached_found

        try:
            # Расширяем запрос для лучшего поиска
            expanded_query = self.expand_query_for_rag(query)

            # Поиск с расширенным запросом
            results = self.rag_system.search(
                query=expanded_query,
                top_k=10,
                similarity_threshold=0.15
            )

            if results:
                logger.info(f"✅ RAG search found {len(results)} results")
                if len(self.rag_cache) < self.cache_size:
                    self.rag_cache[cache_key] = (results, True)
                return results, True
            else:
                # Если ничего не найдено, пробуем с оригинальным запросом
                logger.info("🔄 No results with expanded query, trying original...")
                results = self.rag_system.search(
                    query=query,
                    top_k=10,
                    similarity_threshold=0.1
                )

                if results:
                    logger.info(f"✅ RAG search with original query found {len(results)} results")
                    if len(self.rag_cache) < self.cache_size:
                        self.rag_cache[cache_key] = (results, True)
                    return results, True

                logger.info("❌ RAG search returned no results")
                if len(self.rag_cache) < self.cache_size:
                    self.rag_cache[cache_key] = ([], False)
                return [], False

        except Exception as e:
            logger.error(f"❌ RAG search error: {e}")
            return [], False

    def format_library_response(self, user_message: str, results: List[Dict]) -> Tuple[str, str]:
        """
        Формирует ответ на вопрос о библиотеке на основе результатов RAG
        """
        if not results:
            return None, None

        # Собираем всю информацию из результатов
        information_parts = []
        seen_content = set()

        for i, res in enumerate(results, 1):
            content = res['content'].strip()
            if content in seen_content:
                continue
            seen_content.add(content)

            # Добавляем метаданные
            meta = res['metadata']
            header = f"[Источник {i}: {meta.get('filename', 'unknown')}"
            if meta.get('content_type'):
                header += f", Тип: {meta['content_type']}"
            if meta.get('topics'):
                header += f", Темы: {', '.join(meta['topics'][:3])}"
            if meta.get('age_group'):
                header += f", Возраст: до {meta['age_group']} лет"
            header += "]"

            information_parts.append(f"{header}\n{content}")

        # Объединяем всю информацию
        context_info = "\n\n---\n\n".join(information_parts)

        # Обрезаем если слишком много
        if len(context_info) > 4000:
            first_chunks = information_parts[:5]
            context_info = "\n\n---\n\n".join(first_chunks)
            context_info += "\n\n... (показаны первые 5 результатов)"

        logger.info(f"📊 Context length: {len(context_info)} chars, {len(information_parts)} chunks")

        # Создаем промпт
        library_prompt = self._get_library_prompt()

        # Формируем сообщение с четким выделением вопроса
        user_message_enhanced = f"""ВОПРОС РЕБЁНКА (отвечай ТОЛЬКО на этот вопрос!):
{user_message}

ВЕСЬ КОНТЕКСТ ИЗ БИБЛИОТЕЧНОЙ БАЗЫ (найди в нём ответ на вопрос):
{context_info}

ИНСТРУКЦИЯ:
1. ВНИМАТЕЛЬНО прочитай ВОПРОС РЕБЁНКА
2. Найди в КОНТЕКСТЕ информацию, которая ОТВЕЧАЕТ НА ЭТОТ ВОПРОС
3. Игнорируй информацию из контекста, которая НЕ ОТНОСИТСЯ к вопросу
4. Перескажи найденную информацию простыми словами для ребёнка 7-10 лет
5. Если в контексте нет ответа на вопрос - скажи: "К сожалению, в моей базе нет информации об этом. Лучше спроси у библиотекаря!"
6. Если в контексте есть несколько пунктов - перечисли их все
7. Используй ТОЛЬКО информацию из контекста

ОТВЕТЬ НА ВОПРОС РЕБЁНКА:"""

        return library_prompt, user_message_enhanced

    def filter_response(self, response: str, user_message: str) -> str:
        """
        Проверяет ответ GigaChat на наличие фраз об отсутствии информации
        """
        if not response:
            return response

        # Проверяем, есть ли в ответе фразы об отсутствии информации
        for phrase in self.irrelevant_phrases:
            if re.search(phrase, response.lower()):
                logger.warning(f"⚠️ Response contains irrelevant phrase: {phrase}")
                return self.get_library_redirect_response()

        return response

    def is_library_question(self, text: str) -> bool:
        """Определяет, является ли вопрос вопросом о библиотеке"""
        topic = self.detect_topic(text)
        if topic == TopicCategory.LIBRARY:
            return True

        text_lower = text.lower()
        library_patterns = [
            r'библиотек[аи]?',
            r'книгохранилищ[еа]',
            r'читальн[яю]',
            r'абонемент',
            r'формуляр',
            r'стеллаж',
            r'фонд\s+библиотек',
            r'книговыдач[аи]',
            r'регистраци[яи]',
            r'читател[ьи]',
            r'круж[оек]',
            r'клуб',
            r'проект\s+библиотек',
            r'грант',
            r'интерактивн[ое]?',
            r'расписани[ея]',
            r'график\s+работ',
            r'правил[ао]',
            r'продлени[ея]',
            r'возврат\s+книг'
        ]

        for pattern in library_patterns:
            if re.search(pattern, text_lower):
                return True

        return False

    def get_books_recommendation_prompt(self, text: str) -> str:
        """Формирует запрос для рекомендации книг с наводящими вопросами"""
        topic = self.detect_topic(text)

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

        if topic in questions:
            return questions[topic]

        return "Книги - это здорово! Расскажи, что ты любишь: приключения, фантастику, сказки или что-то другое? Какие книги тебе уже нравились?"

    def get_library_response_with_rag(self, user_message: str) -> Dict:
        """
        Обработка вопроса о библиотеке с использованием RAG
        """
        logger.info(f"🔍 Processing library question with RAG: {user_message[:50]}...")

        # Поиск в RAG с расширением запроса
        results, found = self.search_rag(user_message)

        if found and results:
            logger.info(f"📊 Found {len(results)} results, sending ALL to GigaChat...")

            # Формируем ответ
            library_prompt, enhanced_message = self.format_library_response(user_message, results)

            if library_prompt and enhanced_message:
                return {
                    'messages': [
                        {"role": "system", "content": library_prompt},
                        {"role": "user", "content": enhanced_message}
                    ],
                    'should_override': False,
                    'override_response': None,
                    'rag_used': True,
                    'rag_found': True,
                    'rag_results_count': len(results),
                    'filter_response': True,
                    'user_message': user_message
                }
            else:
                logger.warning("⚠️ Failed to format library response")
                return self._get_fallback_library_response(user_message)
        else:
            logger.info("📋 No RAG results, redirecting to librarian")
            return {
                'messages': [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                'should_override': True,
                'override_response': self.get_library_redirect_response(),
                'rag_used': True,
                'rag_found': False
            }

    def _get_fallback_library_response(self, user_message: str) -> Dict:
        """Fallback ответ, если не удалось сформировать ответ с RAG"""
        return {
            'messages': [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            'should_override': True,
            'override_response': self.get_library_redirect_response(),
            'rag_used': True,
            'rag_found': False
        }

    def get_library_redirect_response(self) -> str:
        """Ответ, когда информация не найдена в RAG или ответ нерелевантный"""
        responses = [
            "Ой, я пока не нашёл в своей базе информации об этом! Но наши библиотекари точно знают ответ. Они сидят за стойкой с книгами - обязательно спроси у них!",
            "Хм, в моей библиотечной базе нет точного ответа на этот вопрос. Но у нас есть замечательные библиотекари - они всегда помогут! Найди их в библиотеке, они в оранжевых бейджиках",
            "К сожалению, я ещё маленький библиотечный помощник и не всё знаю! Давай спросим у настоящего библиотекаря - они очень умные и добрые, точно подскажут!"
        ]
        import random
        return random.choice(responses)

    def get_unsafe_response(self) -> str:
        """Ответ на небезопасный вопрос"""
        responses = [
            "Ой-ой! Это не детский вопрос. Давай лучше поговорим о книгах или космосе! Ты любишь читать?",
            "Хм, это слишком сложная тема для меня. Я всего лишь детский ассистент! Расскажи лучше, что ты читал интересного в последнее время?",
            "Такие вопросы лучше обсуждать с родителями! А я могу предложить тебе книгу о космосе или приключениях. Хочешь?"
        ]
        import random
        return random.choice(responses)

    def get_unknown_topic_response(self) -> str:
        """Ответ, если тема не определена"""
        responses = [
            "Какая интересная тема! Но мне, как детскому ассистенту, лучше рассказывать о книгах, космосе или животных. О чём ты хочешь узнать?",
            "Ух ты! Я пока не совсем понимаю эту тему, но могу рассказать о книгах! Какие истории ты любишь? Приключения или фантастику?",
            "Это звучит увлекательно! А давай поговорим о книгах - я знаю много интересных! Ты любишь читать?"
        ]
        import random
        return random.choice(responses)

    def prepare_messages(self, user_message: str) -> Dict[str, any]:
        """
        Подготавливает сообщения для отправки в GigaChat
        """
        # Проверяем безопасность вопроса
        is_safe, reason = self.is_safe_question(user_message)

        if not is_safe:
            logger.warning(f"⚠️ Unsafe question detected: {user_message[:50]}... Reason: {reason}")
            return {
                'messages': [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                'should_override': True,
                'override_response': self.get_unsafe_response(),
                'rag_used': False,
                'rag_found': False
            }

        # Проверяем, является ли вопрос вопросом о библиотеке
        if self.is_library_question(user_message):
            return self.get_library_response_with_rag(user_message)

        # Если это запрос на рекомендацию книги
        if any(keyword in user_message.lower() for keyword in ['книг', 'посовет', 'рекоменд', 'почитат', 'прочитат']):
            questions = self.get_books_recommendation_prompt(user_message)
            enhanced_message = f"{user_message}\n\nПодсказка для ассистента: Задай наводящие вопросы, чтобы узнать интересы ребёнка. Например: {questions}"

            return {
                'messages': [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": enhanced_message}
                ],
                'should_override': False,
                'override_response': None,
                'rag_used': False,
                'rag_found': False
            }

        # Обычный безопасный запрос
        return {
            'messages': [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            'should_override': False,
            'override_response': None,
            'rag_used': False,
            'rag_found': False
        }

    def validate_response(self, response: str) -> bool:
        """Проверяет, безопасен ли ответ от GigaChat"""
        response_lower = response.lower()

        for keyword in self.unsafe_keywords:
            if keyword in response_lower:
                logger.warning(f"⚠️ Unsafe content in response: {keyword}")
                return False

        return True

    def get_safe_response(self, original_response: str) -> str:
        """Если ответ не прошел проверку, возвращает безопасный ответ"""
        logger.warning("🔄 Replacing unsafe response with safe alternative")
        return "Ой, я немного запутался! Давай лучше поговорим о книгах или о чём-то интересном для детей. Что ты любишь читать?"

    def clear_rag_cache(self):
        """Очищает кэш RAG запросов"""
        self.rag_cache.clear()
        logger.info("🧹 RAG cache cleared")


# Функция для использования в основном приложении
def create_assistant(rag_system=None) -> LibraryAssistant:
    """Создает экземпляр ассистента с опциональной RAG системой"""
    return LibraryAssistant(rag_system=rag_system)