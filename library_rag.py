import os
import pickle
import glob
import re
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LibraryRAGSystem:
    """
    RAG система для библиотечного ассистента.
    Оптимизирована для работы с текстами: правила, мероприятия, клубы, электронные ресурсы.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2',
                 vector_dim: int = 384,
                 chunk_size: int = 800,  # Уменьшил для лучшего захвата конкретной информации
                 chunk_overlap: int = 150):
        """
        Инициализация RAG системы для библиотеки.

        Args:
            model_name: имя модели Sentence-Transformers
            vector_dim: размерность векторов (для all-MiniLM-L6-v2 = 384)
            chunk_size: размер чанка в символах
            chunk_overlap: перекрытие между чанками
        """
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = None
        self.index = None
        self.metadata = []

        # Добавляем кэш для часто задаваемых вопросов
        self.query_cache = {}

    def load_model(self):
        """Загрузка модели Sentence-Transformers."""
        if self.model is None:
            logger.info(f"Загрузка модели {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Модель загружена. Размер эмбеддингов: {self.vector_dim}")

    def extract_metadata_from_text(self, text: str, filename: str) -> Dict:
        """
        Расширенное извлечение мета-тегов из текста для улучшенного поиска.
        """
        metadata = {
            'filename': filename,
            'topics': [],
            'age_group': None,
            'content_type': None,
            'keywords': [],
            'named_entities': [],  # Имена, названия, даты
            'has_dates': False,
            'has_contacts': False
        }

        # ===== ОПРЕДЕЛЕНИЕ ТИПА КОНТЕНТА =====
        content_patterns = {
            'events': [
                r'(мероприяти[ея]|квартирник|библионочь|лекци[яи]|встреч[ау]|праздник|фестиваль|конкурс|акция|неделя|день|час|урок|экскурсия|познавательный|игровой)',
                r'\d{1,2}\s+(январ[яь]|феврал[яь]|март[а]?|апрел[яь]|ма[яй]|июн[яь]|июл[яь]|август[а]?|сентябр[яь]|октябр[яь]|ноябр[яь]|декабр[яь]|января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)'
            ],
            'clubs': [
                r'(клуб|объединени[ея]|круж[оек]|студия|школа|мастерская|академия|театр|мастер-класс|занятие|обучение)'
            ],
            'rules': [
                r'(правил[ао]|пользовани[ею]|задолженност[ьи]|просрочк[ау]|запись|формуляр|возврат|читательский|должник|замена|утрата|повреждение)'
            ],
            'resources': [
                r'(электронн[ыy]|ресурс[аы]|литрес|НЭБ|баз[ау]|журнал[аы]|каталог|сайт|портал|онлайн|доступ|цифровой|интерактивный)'
            ],
            'about_library': [
                r'(история|основан|открыт|год|юбилей|проект|грант|пространство|оборудование|структура|отдел|абонемент|читальный зал)'
            ]
        }

        for content_type, patterns in content_patterns.items():
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
                if metadata['content_type'] is None:
                    metadata['content_type'] = content_type
                elif content_type != 'about_library' and metadata['content_type'] == 'about_library':
                    metadata['content_type'] = content_type  # Приоритет у специфических типов

        # ===== ИЗВЛЕЧЕНИЕ ВОЗРАСТНЫХ МЕТОК =====
        age_patterns = [
            r'(\d+)\s*(лет|год|года|age|years?)\s*[+-]?',
            r'(до|с|от)\s*(\d+)\s*лет',
            r'для\s*(детей|подростков)\s*(до|с|от)\s*(\d+)\s*лет'
        ]

        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                age = match.group(1) if 'до' in pattern or 'с' in pattern or 'от' in pattern else match.group(1)
                metadata['age_group'] = age
                break

        # ===== ИЗВЛЕЧЕНИЕ ИМЕН И НАЗВАНИЙ =====
        # Названия проектов, клубов, мероприятий в кавычках
        entities = re.findall(r'["«]([^"»]+)["»]', text)
        if entities:
            metadata['named_entities'].extend(entities[:5])

        # Названия с большой буквы (простейшая эвристика)
        capitalized = re.findall(r'\b([А-Я][а-я]+(?:\s+[А-Я][а-я]+)*)\b', text)
        if capitalized:
            metadata['named_entities'].extend(capitalized[:3])

        # ===== ПОИСК ДАТ =====
        date_patterns = [
            r'\d{1,2}\s+(январ[яь]|феврал[яь]|март[а]?|апрел[яь]|ма[яй]|июн[яь]|июл[яь]|август[а]?|сентябр[яь]|октябр[яь]|ноябр[яь]|декабр[яь])',
            r'\d{2}\.\d{2}\.\d{4}',
            r'\d{4}\s+год[ау]?',
            r'\d{1,2}\s+числа'
        ]
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in date_patterns):
            metadata['has_dates'] = True

        # ===== ПОИСК КОНТАКТОВ =====
        if re.search(r'(тел|phone|контакт|@|\.ru|\.com)', text, re.IGNORECASE):
            metadata['has_contacts'] = True

        # ===== РАСШИРЕННЫЕ ТЕМЫ =====
        topics_keywords = {
            'музыка': ['музыкальн', 'квартирник', 'песни', 'гитара', 'концерт'],
            'спорт': ['настольные игры', 'шахматы', 'турнир', 'спорт', 'движение'],
            'наука': ['лекция', 'научн', 'познавательн', 'астрономия', 'опыт', 'эксперимент'],
            'детство': ['дет', 'ребёнк', 'малыш', 'родитель', 'семья', 'дошкольник'],
            'история': ['историческ', 'реконструкц', 'краеведен', 'прошлое', 'память'],
            'языки': ['английск', 'English', 'разговорн', 'бурятск', 'язык'],
            'творчество': ['творческ', 'рисован', 'лепк', 'поделк', 'мастер', 'рукоделие'],
            'экология': ['экологическ', 'природ', 'нерпёнок', 'лесовичок', 'экосистем', 'защит'],
            'литература': ['книг', 'чтени', 'писател', 'поэт', 'сказк', 'рассказ', 'роман'],
            'технологии': ['интерактивн', 'цифров', 'виртуальн', 'мультфильм', 'оборудование']
        }

        for topic, keywords in topics_keywords.items():
            if any(kw in text.lower() for kw in keywords):
                metadata['topics'].append(topic)
                metadata['keywords'].extend(keywords[:2])

        # Удаляем дубликаты
        metadata['keywords'] = list(set(metadata['keywords']))[:5]
        metadata['topics'] = list(set(metadata['topics']))

        return metadata

    def chunk_text(self, text: str, filename: str, chunk_id: int) -> List[Dict]:
        """
        Улучшенное разбиение текста на чанки с учетом структуры документа.
        """
        chunks = []

        # Очистка текста
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Нормализация переносов
        text = re.sub(r'[ \t]+', ' ', text)  # Удаление лишних пробелов

        # Разбиение по логическим блокам (заголовки, абзацы)
        # Ищем заголовки (текст с двоеточием в конце или отдельный абзац)
        lines = text.split('\n')
        paragraphs = []
        current_para = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_para:
                    paragraphs.append(' '.join(current_para))
                    current_para = []
                continue

            # Если строка заканчивается на ":" и не слишком длинная - это заголовок
            if line.endswith(':') and len(line) < 100 and current_para:
                paragraphs.append(' '.join(current_para))
                current_para = [line]
            else:
                current_para.append(line)

        if current_para:
            paragraphs.append(' '.join(current_para))

        # Объединяем слишком короткие абзацы
        merged_paragraphs = []
        for para in paragraphs:
            if len(para) < 100 and merged_paragraphs and len(merged_paragraphs[-1]) < self.chunk_size * 0.7:
                merged_paragraphs[-1] += ' ' + para
            else:
                merged_paragraphs.append(para)

        # Создание чанков
        current_chunk = ""
        for para in merged_paragraphs:
            para_len = len(para)

            # Если абзац очень длинный - жесткое разбиение
            if para_len > self.chunk_size * 1.5:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_id': chunk_id,
                        'type': 'paragraph'
                    })
                    chunk_id += 1
                    current_chunk = ""

                # Разбиваем по предложениям
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = ""
                for sent in sentences:
                    if len(temp_chunk) + len(sent) < self.chunk_size:
                        temp_chunk += sent + ' '
                    else:
                        if temp_chunk:
                            chunks.append({
                                'text': temp_chunk.strip(),
                                'chunk_id': chunk_id,
                                'type': 'sentence'
                            })
                            chunk_id += 1
                        temp_chunk = sent + ' '
                if temp_chunk:
                    chunks.append({
                        'text': temp_chunk.strip(),
                        'chunk_id': chunk_id,
                        'type': 'sentence'
                    })
                    chunk_id += 1

            # Обычный абзац
            elif len(current_chunk) + para_len + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += ' ' + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_id': chunk_id,
                        'type': 'paragraph'
                    })
                    chunk_id += 1
                current_chunk = para

        # Добавляем последний чанк
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': chunk_id,
                'type': 'paragraph'
            })
            chunk_id += 1

        # Постобработка: если чанки слишком маленькие, объединяем их
        optimized_chunks = []
        i = 0
        while i < len(chunks):
            if i < len(chunks) - 1 and len(chunks[i]['text']) < 200:
                combined = chunks[i]['text'] + ' ' + chunks[i + 1]['text']
                if len(combined) <= self.chunk_size:
                    optimized_chunks.append({
                        'text': combined,
                        'chunk_id': chunks[i]['chunk_id'],
                        'type': 'combined'
                    })
                    i += 2
                    continue
            optimized_chunks.append(chunks[i])
            i += 1

        return optimized_chunks, chunk_id

    def train(self, data_dir: str = './data', save_dir: str = 'vector_store'):
        """
        Обучение RAG: векторизация всех текстов из ./data/text*.txt
        """
        self.load_model()

        # Поиск всех текстовых файлов
        txt_files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))

        # Если нет .txt, ищем любые текстовые файлы
        if not txt_files:
            txt_files = sorted(glob.glob(os.path.join(data_dir, '*.md'))) + \
                        sorted(glob.glob(os.path.join(data_dir, '*.rst')))

        if not txt_files:
            raise ValueError(f"❌ Не найдено текстовых файлов в директории {data_dir}")

        logger.info(f"Найдено {len(txt_files)} файлов: {[os.path.basename(f) for f in txt_files]}")

        all_chunks = []
        self.metadata = []
        global_chunk_id = 0

        for file_idx, filepath in enumerate(txt_files, 1):
            filename = os.path.basename(filepath)
            logger.info(f"Обработка {filename}...")

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    full_text = f.read().strip()
            except UnicodeDecodeError:
                # Пробуем другие кодировки
                with open(filepath, 'r', encoding='cp1251') as f:
                    full_text = f.read().strip()

            if not full_text:
                logger.warning(f"Файл {filename} пуст, пропускаем")
                continue

            # Извлечение метаданных
            file_metadata = self.extract_metadata_from_text(full_text, filename)

            # Разбиение на чанки
            chunks, global_chunk_id = self.chunk_text(full_text, filename, global_chunk_id)
            logger.info(f"  Разбито на {len(chunks)} чанков")

            # Сохранение чанков
            for chunk in chunks:
                all_chunks.append(chunk['text'])
                self.metadata.append({
                    'file_idx': file_idx,
                    'filename': filename,
                    'filepath': filepath,
                    'chunk_id': chunk['chunk_id'],
                    'chunk_type': chunk['type'],
                    'text': chunk['text'],
                    'length_chars': len(chunk['text']),
                    'length_words': len(chunk['text'].split()),
                    'content_type': file_metadata['content_type'],
                    'topics': file_metadata['topics'],
                    'age_group': file_metadata['age_group'],
                    'keywords': file_metadata['keywords'],
                    'named_entities': file_metadata['named_entities'],
                    'has_dates': file_metadata['has_dates'],
                    'has_contacts': file_metadata['has_contacts']
                })

        logger.info(f"Итого: {len(all_chunks)} чанков для векторизации")

        # Векторизация с учетом длинных текстов
        logger.info("Векторизация текстов...")
        embeddings = self.model.encode(
            all_chunks,
            show_progress_bar=True,
            batch_size=32
        )
        embeddings = np.array(embeddings).astype('float32')

        # Создание FAISS индекса
        self.index = faiss.IndexFlatIP(self.vector_dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        # Сохранение
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, 'library_index.faiss'))

        with open(os.path.join(save_dir, 'library_metadata.pkl'), 'wb') as f:
            pickle.dump(self.metadata, f)

        logger.info(f"Векторная база сохранена в {save_dir}")
        logger.info(f"  - Индекс: {self.index.ntotal} векторов")
        logger.info(f"  - Метаданные: {len(self.metadata)} записей")

    def load(self, store_dir: str = 'vector_store'):
        """Загрузка существующей векторной базы."""
        index_path = os.path.join(store_dir, 'library_index.faiss')
        metadata_path = os.path.join(store_dir, 'library_metadata.pkl')

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"❌ База не найдена в {store_dir}.")

        self.load_model()
        self.index = faiss.read_index(index_path)

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        logger.info(f"Загружена база: {self.index.ntotal} векторов, {len(self.metadata)} чанков")

    def search(self, query: str, top_k: int = 5,
               similarity_threshold: float = 0.4,
               filter_by_type: Optional[str] = None,
               filter_by_topic: Optional[str] = None,
               filter_age_group: Optional[str] = None) -> List[Dict]:
        """
        Расширенный поиск с фильтрацией.
        """
        if self.index is None:
            raise RuntimeError("❌ База не загружена.")

        # Проверка кэша
        cache_key = f"{query}_{top_k}_{similarity_threshold}_{filter_by_type}_{filter_by_topic}_{filter_age_group}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]

        # Векторизация запроса
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Поиск (запрашиваем больше для фильтрации)
        search_k = min(top_k * 5, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < similarity_threshold:
                continue

            meta = self.metadata[idx]

            # Фильтрация по типу
            if filter_by_type and meta['content_type'] != filter_by_type:
                continue

            # Фильтрация по теме
            if filter_by_topic and filter_by_topic not in meta['topics']:
                continue

            # Фильтрация по возрастной группе
            if filter_age_group and meta['age_group']:
                try:
                    if int(meta['age_group']) > int(filter_age_group):
                        continue
                except (ValueError, TypeError):
                    pass

            results.append({
                'similarity_score': float(score),
                'metadata': meta,
                'content': meta['text']
            })

            if len(results) >= top_k:
                break

        # Кэширование результата
        self.query_cache[cache_key] = results
        return results

    def get_context_for_gigachat(self, query: str, top_k: int = 3,
                                 similarity_threshold: float = 0.35,
                                 max_chars: int = 3000,
                                 include_metadata: bool = True) -> str:
        """
        Формирование контекста для GigaChat с улучшенной структурой.
        """
        results = self.search(query, top_k, similarity_threshold)

        if not results:
            return "КОНТЕКСТ: В библиотечной базе не найдено релевантной информации по данному запросу."

        # Группировка результатов по темам
        grouped_results = {}
        for res in results:
            topic = 'general'
            if res['metadata']['topics']:
                topic = res['metadata']['topics'][0]
            if topic not in grouped_results:
                grouped_results[topic] = []
            grouped_results[topic].append(res)

        context_parts = []
        total_chars = 0

        # Сначала добавляем наиболее релевантные результаты
        for topic, items in grouped_results.items():
            # Сортируем внутри группы по релевантности
            items.sort(key=lambda x: x['similarity_score'], reverse=True)

            for res in items[:2]:  # Берем не более 2 из каждой темы
                meta = res['metadata']
                score = res['similarity_score']

                if include_metadata:
                    header_parts = [
                        f"📄 Источник: {meta['filename']}",
                        f"📊 Релевантность: {score:.2f}"
                    ]
                    if meta['content_type']:
                        header_parts.append(f"🏷️ Тип: {meta['content_type']}")
                    if meta['topics']:
                        header_parts.append(f"🔖 Темы: {', '.join(meta['topics'][:3])}")
                    if meta['age_group']:
                        header_parts.append(f"👤 Возраст: до {meta['age_group']} лет")

                    header = " | ".join(header_parts) + "\n" + "-" * 40 + "\n"
                else:
                    header = ""

                chunk_text = res['content']

                # Обрезка по длине с сохранением смысла
                if total_chars + len(header) + len(chunk_text) > max_chars:
                    remaining = max_chars - total_chars - len(header) - 100
                    if remaining > 200:
                        # Обрезаем по предложению
                        sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
                        trimmed = ""
                        for sent in sentences:
                            if len(trimmed) + len(sent) < remaining:
                                trimmed += sent + ' '
                            else:
                                break
                        chunk_text = trimmed.strip() + "...\n[Контекст обрезан по длине]"
                    else:
                        break

                context_parts.append(header + chunk_text)
                total_chars += len(header) + len(chunk_text)

        return "🏛️ КОНТЕКСТ ДЛЯ БИБЛИОТЕЧНОГО АССИСТЕНТА:\n\n" + "\n\n---\n\n".join(context_parts)

    def get_categorized_context(self, query: str, top_k: int = 5) -> Dict[str, List[Dict]]:
        """
        Возвращает контекст, сгруппированный по типам контента.
        """
        results = self.search(query, top_k, similarity_threshold=0.3)

        categorized = {
            'events': [],
            'clubs': [],
            'rules': [],
            'resources': [],
            'about_library': [],
            'other': []
        }

        for res in results:
            content_type = res['metadata']['content_type'] or 'other'
            if content_type in categorized:
                categorized[content_type].append(res)
            else:
                categorized['other'].append(res)

        return categorized

    def clear_cache(self):
        """Очистка кэша запросов."""
        self.query_cache = {}
        logger.info("Кэш очищен")


# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========

def main():
    """
    Пример использования улучшенной RAG системы для библиотеки.
    """
    # Инициализация
    rag = LibraryRAGSystem(
        model_name='all-MiniLM-L6-v2',
        vector_dim=384,
        chunk_size=800,  # Уменьшенный размер для более точного поиска
        chunk_overlap=150
    )

    print("\n" + "=" * 60)
    print("📚 БИБЛИОТЕЧНАЯ RAG СИСТЕМА (УЛУЧШЕННАЯ)")
    print("=" * 60)

    # Проверяем наличие данных
    DATA_PATH = "./data"

    if os.path.exists(DATA_PATH):
        rag.train(data_dir=DATA_PATH, save_dir='vector_store')
    else:
        logger.warning(f"Директория {DATA_PATH} не найдена. Создаю тестовую...")
        os.makedirs(DATA_PATH, exist_ok=True)

        # Пример текста из вашего файла
        test_content = """Детская библиотека (библиотека-филиал №17) была основана в 1966 году. 
        В 1968 году ей было присвоено имя детского писателя Аркадия Петровича Гайдара.

        Библиотека обслуживает детей до 14 лет включительно, а также их родителей, учителей.

        Структура библиотеки:
        - Младший абонемент;
        - Старший абонемент;
        - Читальный зал с интерактивным оборудованием.

        В библиотеке работают кружки и клубы:
        Клуб народной куклы "Куклеюшка".
        Кукольный театр "Алтан гэрхэн".
        Кружок экологического конструирования "Лесовичок".
        Литературно-драматический кружок "Радуга".
        Экологический кружок "Нерпёнок".
        """

        with open(os.path.join(DATA_PATH, 'text1.txt'), 'w', encoding='utf-8') as f:
            f.write(test_content)

        rag.train(data_dir=DATA_PATH, save_dir='vector_store')

    # Загрузка и тестирование
    rag.load(save_dir='vector_store')

    print("\n" + "=" * 60)
    print("🔍 ТЕСТИРОВАНИЕ ПОИСКА")
    print("=" * 60)

    test_queries = [
        "Какие есть кружки для детей?",
        "Сколько лет библиотеке?",
        "Что такое Нерпёнок?",
        "Есть ли кукольный театр?"
    ]

    for query in test_queries:
        print(f"\n📌 ЗАПРОС: {query}")
        print("-" * 40)

        context = rag.get_context_for_gigachat(
            query=query,
            top_k=3,
            similarity_threshold=0.3
        )

        print(context[:800] + "..." if len(context) > 800 else context)
        print("\n" + "=" * 60)

        # Показать категоризацию
        categorized = rag.get_categorized_context(query, top_k=3)
        print("\n📊 Категоризация результатов:")
        for cat, items in categorized.items():
            if items:
                print(f"  {cat}: {len(items)} результатов")


if __name__ == "__main__":
    main()