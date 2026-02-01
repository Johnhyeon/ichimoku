"""환경변수 설정 로더"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
env_path = Path(__file__).parent.parent / "config" / ".env"
load_dotenv(env_path)


class Settings:
    """설정 클래스"""

    # 바이빗 API
    BYBIT_API_KEY: str = os.getenv("BYBIT_API_KEY", "")
    BYBIT_SECRET: str = os.getenv("BYBIT_SECRET", "")
    BYBIT_TESTNET: bool = os.getenv("BYBIT_TESTNET", "false").lower() == "true"

    # 페이퍼 트레이딩
    PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"
    PAPER_INITIAL_BALANCE: float = float(os.getenv("PAPER_INITIAL_BALANCE", "1000"))

    # 텔레그램
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Google Gemini
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")


settings = Settings()
