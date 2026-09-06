# gigachat_client.py
import time
import uuid
import logging
import requests
import base64
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class GigaChatClient:
    """Клиент для работы с GigaChat API"""

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expires_at = 0
        self.auth_url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
        self.api_url = "https://gigachat.devices.sberbank.ru/api/v1"

        logger.info(f"GigaChat client initialized with client_id: {client_id[:10]}...")

    def _generate_rquid(self) -> str:
        return str(uuid.uuid4())

    def get_access_token(self) -> str:
        """Получение access token"""
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': self._generate_rquid(),
        }

        # Basic авторизация
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode('utf-8')).decode('utf-8')
        headers['Authorization'] = f'Basic {encoded_credentials}'

        payload = {'scope': 'GIGACHAT_API_PERS'}

        try:
            logger.info("Requesting GigaChat access token...")
            response = requests.post(self.auth_url, headers=headers, data=payload, verify=False, timeout=30)

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 1800)
                self.token_expires_at = time.time() + expires_in
                logger.info(f"✅ GigaChat token obtained, expires in {expires_in}s")
                return self.access_token
            else:
                logger.error(f"Auth failed: {response.status_code} - {response.text}")
                raise Exception(f"GigaChat auth failed: {response.text}")

        except Exception as e:
            logger.error(f"GigaChat auth error: {e}")
            raise

    def _is_token_valid(self) -> bool:
        return self.access_token is not None and time.time() < self.token_expires_at - 60

    def _ensure_valid_token(self):
        if not self._is_token_valid():
            logger.info("Token expired, getting new one...")
            self.get_access_token()

    def chat(self, messages: List[Dict[str, str]],
             temperature: float = 0.7,
             max_tokens: int = 2000) -> Optional[str]:
        """Отправка запроса к GigaChat"""
        try:
            self._ensure_valid_token()

            url = f"{self.api_url}/chat/completions"

            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
                'Authorization': f'Bearer {self.access_token}'
            }

            payload = {
                'model': 'GigaChat',
                'messages': messages,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'stream': False,
                'repetition_penalty': 1.0,
            }

            logger.info(f"Sending request to GigaChat...")
            response = requests.post(url, headers=headers, json=payload, verify=False, timeout=60)

            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                logger.info(f"✅ GigaChat response received, length: {len(answer)} chars")
                return answer
            elif response.status_code == 401:
                logger.warning("Token expired during request, refreshing...")
                self.get_access_token()
                return self.chat(messages, temperature, max_tokens)
            else:
                logger.error(f"GigaChat error: {response.status_code} - {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error("GigaChat request timeout")
            return None
        except Exception as e:
            logger.error(f"GigaChat request error: {e}")
            return None

    def simple_chat(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """Упрощённый метод для одного запроса"""
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, temperature=temperature)


# Тестовая функция
def test_gigachat():
    """Тестирование подключения к GigaChat"""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    client_id = os.getenv('GIGACHAT_CLIENT_ID')
    client_secret = os.getenv('GIGACHAT_CLIENT_SECRET')

    if not client_id or not client_secret:
        print("❌ GIGACHAT_CLIENT_ID и GIGACHAT_CLIENT_SECRET не заданы в .env")
        return False

    print(f"Testing GigaChat with client_id: {client_id[:10]}...")

    try:
        client = GigaChatClient(client_id, client_secret)
        token = client.get_access_token()
        print(f"✅ Token obtained: {token[:50]}...")

        # Тестовый запрос
        response = client.simple_chat("Скажи 'Привет, Тимурка!' на русском языке")
        print(f"✅ Test response: {response}")
        return True

    except Exception as e:
        print(f"❌ GigaChat test failed: {e}")
        return False


if __name__ == "__main__":
    # Отключаем SSL warnings
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    test_gigachat()