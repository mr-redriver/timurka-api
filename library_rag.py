import os
import pickle
import glob
import re
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


class LibraryRAGSystem:
    """
    RAG система для библиотечного ассистента.
    Оптимизирована для работы с текстами: правила, мероприятия, клубы, электронные ресурсы.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2',
                 vector_dim: int = 384,
                 chunk_size: int = 1500,
                 chunk_overlap: int = 200):
        """
        Инициализация RAG системы для библиотеки.

        Args:
            model_name: имя модели Sentence-Transformers
            vector_dim: размерность векторов (для all-MiniLM-L6-v2 = 384)
            chunk_size: размер чанка в символах (для длинных текстов)
            chunk_overlap: перекрытие между чанками
        """
        self.model_name = model_name
        self.vector_dim = vector_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = None
        self.index = None
        self.metadata = []  # список словарей с метаданными для каждого чанка

    def load_model(self):
        """Загрузка модели Sentence-Transformers."""
        if self.model is None:
            print(f"📚 Загрузка модели {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Модель загружена. Размер эмбеддингов: {self.vector_dim}")

    def extract_metadata_from_text(self, text: str, filename: str) -> Dict:
        """
        Извлечение мета-тегов из текста для улучшенного поиска.
        Ищет ключевые слова: тематика, возраст, тип контента.

        Args:
            text: текст содержимого
            filename: имя файла

        Returns:
            словарь с извлеченными метаданными
        """
        metadata = {
            'filename': filename,
            'topics': [],
            'age_group': None,
            'content_type': None,  # 'events', 'clubs', 'rules', 'resources', 'recommendations'
            'keywords': []
        }

        # Определение типа контента по ключевым словам
        if re.search(r'(мероприяти[ея]|квартирник|библионочь|лекци[яи]|встреч[ау])', text, re.IGNORECASE):
            metadata['content_type'] = 'events'
        elif re.search(r'(клуб|объединени[ея]|круж[оек]|English Corner)', text, re.IGNORECASE):
            metadata['content_type'] = 'clubs'
        elif re.search(r'(правил[ао]|пользовани[ею]|задолженност[ьи]|просрочк[ау])', text, re.IGNORECASE):
            metadata['content_type'] = 'rules'
        elif re.search(r'(электронн[ыy]|ресурс[аы]|литрес|НЭБ|баз[ау]|журнал[аы])', text, re.IGNORECASE):
            metadata['content_type'] = 'resources'
        elif re.search(r'(рекомендаци[ия]|книг[аи]|прочитать|почитать|список)', text, re.IGNORECASE):
            metadata['content_type'] = 'recommendations'

        # Поиск возрастных меток
        age_match = re.search(r'(\d+)\s*(лет|год|года|age|years?)\s*[+-]?', text, re.IGNORECASE)
        if age_match:
            metadata['age_group'] = age_match.group(1)

        # Извлечение ключевых слов (простых тематик)
        topics_keywords = {
            'музыка': ['музыкальный', 'квартирник', 'песни', 'гитара'],
            'спорт': ['настольные игры', 'шахматы', 'турнир'],
            'наука': ['лекция', 'научный', 'познавательный', 'астрономия'],
            'детство': ['дет', 'ребёнок', 'малыш', 'родитель'],
            'история': ['историческ', 'реконструкц', 'краеведен'],
            'языки': ['английский', 'English', 'разговорный']
        }

        for topic, keywords in topics_keywords.items():
            if any(kw in text.lower() for kw in keywords):
                metadata['topics'].append(topic)
                metadata['keywords'].extend(keywords[:2])

        return metadata

    def chunk_text(self, text: str, filename: str, chunk_id: int) -> List[Dict]:
        """
        Умное разбиение текста на чанки с сохранением границ абзацев.

        Args:
            text: исходный текст
            filename: имя файла
            chunk_id: базовый идентификатор чанка

        Returns:
            список чанков с метаданными
        """
        chunks = []

        # Сначала пробуем разбить по двойным переносам строк (абзацы)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        current_chunk = ""
        current_length = 0

        for para in paragraphs:
            para_len = len(para)

            # Если абзац сам по себе больше chunk_size, разбиваем его жестко
            if para_len > self.chunk_size:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_id': chunk_id,
                        'type': 'paragraph'
                    })
                    chunk_id += 1
                    current_chunk = ""
                    current_length = 0

                # Жесткое разбиение длинного абзаца
                for i in range(0, para_len, self.chunk_size - self.chunk_overlap):
                    chunk_text = para[i:i + self.chunk_size]
                    chunks.append({
                        'text': chunk_text.strip(),
                        'chunk_id': chunk_id,
                        'type': 'forced'
                    })
                    chunk_id += 1

            # Если добавление абзаца не превышает лимит - добавляем
            elif current_length + para_len + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_length += para_len + 2

            # Иначе сохраняем текущий чанк и начинаем новый
            else:
                if current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'chunk_id': chunk_id,
                        'type': 'paragraph'
                    })
                    chunk_id += 1
                current_chunk = para
                current_length = para_len

        # Добавляем последний чанк
        if current_chunk:
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': chunk_id,
                'type': 'paragraph'
            })
            chunk_id += 1

        return chunks, chunk_id

    def train(self, data_dir: str = './data', save_dir: str = 'vector_store'):
        """
        Обучение RAG: векторизация всех текстов из ./data/text*.txt

        Args:
            data_dir: директория с файлами text1.txt, text2.txt и т.д.
            save_dir: директория для сохранения векторной базы
        """
        self.load_model()

        # Поиск всех файлов text*.txt
        txt_files = sorted(glob.glob(os.path.join(data_dir, 'text*.txt')))

        if not txt_files:
            raise ValueError(f"❌ Не найдено файлов text*.txt в директории {data_dir}")

        print(f"📁 Найдено {len(txt_files)} файлов: {[os.path.basename(f) for f in txt_files]}")

        all_chunks = []  # список текстов чанков
        self.metadata = []  # метаданные для каждого чанка
        global_chunk_id = 0

        # Обработка каждого файла
        for file_idx, filepath in enumerate(txt_files, 1):
            filename = os.path.basename(filepath)
            print(f"\n📄 Обработка {filename}...")

            with open(filepath, 'r', encoding='utf-8') as f:
                full_text = f.read().strip()

            if not full_text:
                print(f"  ⚠️ Файл пуст, пропускаем")
                continue

            # Извлечение метаданных из всего текста
            file_metadata = self.extract_metadata_from_text(full_text, filename)

            # Разбиение на чанки
            chunks, global_chunk_id = self.chunk_text(full_text, filename, global_chunk_id)

            print(f"  ✂️ Разбито на {len(chunks)} чанков (размер чанка: {self.chunk_size} симв.)")

            # Добавление чанков в общий список
            for chunk in chunks:
                all_chunks.append(chunk['text'])

                # Метаданные для каждого чанка
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
                    'keywords': file_metadata['keywords']
                })

        print(f"\n📊 Итого: {len(all_chunks)} чанков для векторизации")

        # Векторизация
        print("🔄 Векторизация текстов...")
        embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')

        # Создание FAISS индекса (используем косинусное сходство через нормализацию)
        self.index = faiss.IndexFlatIP(self.vector_dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        # Сохранение
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, 'library_index.faiss'))

        with open(os.path.join(save_dir, 'library_metadata.pkl'), 'wb') as f:
            pickle.dump(self.metadata, f)

        print(f"\n✅ Векторная база сохранена в {save_dir}")
        print(f"   - Индекс: {self.index.ntotal} векторов")
        print(f"   - Метаданные: {len(self.metadata)} записей")

    def load(self, store_dir: str = 'vector_store'):
        """
        Загрузка существующей векторной базы.

        Args:
            store_dir: директория с сохраненной базой
        """
        index_path = os.path.join(store_dir, 'library_index.faiss')
        metadata_path = os.path.join(store_dir, 'library_metadata.pkl')

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"❌ База не найдена в {store_dir}. Сначала выполните обучение.")

        self.load_model()
        self.index = faiss.read_index(index_path)

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

        print(f"✅ Загружена база: {self.index.ntotal} векторов, {len(self.metadata)} чанков")

    def search(self, query: str, top_k: int = 5,
               similarity_threshold: float = 0.4,
               filter_by_type: Optional[str] = None) -> List[Dict]:
        """
        Поиск релевантных чанков по запросу.

        Args:
            query: текстовый запрос пользователя
            top_k: количество возвращаемых результатов
            similarity_threshold: минимальный порог схожести (0-1)
            filter_by_type: опциональная фильтрация ('events', 'clubs', 'rules', 'resources', 'recommendations')

        Returns:
            список результатов с метаданными и оценкой
        """
        if self.index is None:
            raise RuntimeError("❌ База не загружена. Вызовите load() или train() сначала.")

        # Векторизация запроса
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)

        # Поиск (запрашиваем больше для возможной фильтрации)
        search_k = top_k * 3 if filter_by_type else top_k
        scores, indices = self.index.search(query_embedding, min(search_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or score < similarity_threshold:
                continue

            # Фильтрация по типу контента
            if filter_by_type and self.metadata[idx]['content_type'] != filter_by_type:
                continue

            results.append({
                'similarity_score': float(score),
                'metadata': self.metadata[idx],
                'content': self.metadata[idx]['text']
            })

            if len(results) >= top_k:
                break

        return results

    def get_context_for_gigachat(self, query: str, top_k: int = 3,
                                 similarity_threshold: float = 0.4,
                                 max_chars: int = 3000) -> str:
        """
        Формирование контекста для передачи в GigaChat API.

        Args:
            query: запрос пользователя
            top_k: количество чанков
            similarity_threshold: порог схожести
            max_chars: максимальная длина контекста

        Returns:
            строка-контекст для LLM
        """
        results = self.search(query, top_k, similarity_threshold)

        if not results:
            return "Контекст: В библиотечной базе не найдено релевантной информации по данному запросу."

        context_parts = []
        total_chars = 0

        for i, res in enumerate(results, 1):
            meta = res['metadata']
            score = res['similarity_score']

            # Формируем заголовок с мета-информацией
            header = f"[Источник: {meta['filename']}"
            if meta['content_type']:
                header += f" | Тип: {meta['content_type']}"
            if meta['topics']:
                header += f" | Темы: {', '.join(meta['topics'])}"
            header += f" | Релевантность: {score:.2f}]\n"

            chunk_text = res['content']

            # Ограничение по длине
            if total_chars + len(header) + len(chunk_text) > max_chars:
                remaining = max_chars - total_chars - len(header) - 50
                if remaining > 200:
                    chunk_text = chunk_text[:remaining] + "...\n[Контекст обрезан по длине]"
                else:
                    break

            context_parts.append(header + chunk_text)
            total_chars += len(header) + len(chunk_text)

        return "КОНТЕКСТ ДЛЯ БИБЛИОТЕЧНОГО АССИСТЕНТА:\n\n" + "\n\n---\n\n".join(context_parts)


# ========== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==========

def main():
    """
    Пример использования RAG системы для библиотеки.
    """

    # Инициализация
    rag = LibraryRAGSystem(
        model_name='all-MiniLM-L6-v2',
        vector_dim=384,
        chunk_size=1200,  # Чанки по 1200 символов для библиотечных текстов
        chunk_overlap=200
    )

    # ===== РЕЖИМ 1: ОБУЧЕНИЕ (однократно) =====
    print("\n" + "=" * 60)
    print("📚 БИБЛИОТЕЧНАЯ RAG СИСТЕМА - РЕЖИМ ОБУЧЕНИЯ")
    print("=" * 60)

    # Укажите ваш путь к папке ./data с файлами text1.txt, text2.txt и т.д.
    DATA_PATH = "./data"  # Измените на ваш абсолютный путь, если нужно
    # Например: DATA_PATH = "C:/Users/YourName./data" или DATA_PATH = "../data"

    if os.path.exists(DATA_PATH):
        rag.train(data_dir=DATA_PATH, save_dir='vector_store')
    else:
        print(f"⚠️ Директория {DATA_PATH} не найдена.")
        print("Создаю тестовую директорию ./test_data для демонстрации...")

        # Демонстрация на тестовых данных
        os.makedirs('./test_data', exist_ok=True)

        # Сохраняем сгенерированные вами тексты как example
        test_files = [
            ('text1.txt',
             "Музыкальный квартирник состоится 29 мая в 19:00. Приходите с гитарами и хорошим настроением..."),
            ('text2.txt', "Библионочь 2026 пройдет 6 июня. Тема: Традиции будущего. Вход свободный..."),
            ('text3.txt', "Клуб любителей фантастики 'Терра Инкогнита' собирается каждую субботу в 15:00..."),
        ]

        for fname, content in test_files:
            with open(f'./test_data/{fname}', 'w', encoding='utf-8') as f:
                f.write(content)

        rag.train(data_dir='./test_data', save_dir='vector_store')

    # ===== РЕЖИМ 2: ПОИСК И КОНТЕКСТ =====
    print("\n" + "=" * 60)
    print("🔍 БИБЛИОТЕЧНАЯ RAG СИСТЕМА - ПОИСК")
    print("=" * 60)

    rag.load(save_dir='vector_store')

    # Тестовые запросы (эмулирующие вопросы читателей)
    test_queries = [
        "Когда будет ближайший квартирник?",
        "Какие есть клубы для любителей фантастики?",
        "Расскажи про электронные ресурсы библиотеки",
        "Что можно почитать семилетнему ребенку?",
        "Правила продления книги онлайн"
    ]

    for query in test_queries:
        print(f"\n📌 ЗАПРОС: {query}")
        print("-" * 40)

        # Поиск с фильтрацией (пример)
        context = rag.get_context_for_gigachat(
            query=query,
            top_k=3,
            similarity_threshold=0.35
        )

        print(context[:500] + "..." if len(context) > 500 else context)
        print("\n" + "=" * 60)

        # Дополнительно: вывод результатов с метаданными
        print("\n📊 Детальные результаты поиска:")
        results = rag.search(query, top_k=2)
        for i, res in enumerate(results, 1):
            print(
                f"  {i}. {res['metadata']['filename']} (релевантность: {res['similarity_score']:.3f}) - тип: {res['metadata']['content_type']}")


if __name__ == "__main__":
    main()