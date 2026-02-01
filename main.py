#!/usr/bin/env python3
"""
Ichimoku Cloud 자동매매 봇

실행 예시:
    python main.py                 # 메인넷 LIVE (주의: 실제 주문)
    python main.py --paper         # 페이퍼 모드 (주문 안 나감)
    python main.py --testnet       # 테스트넷
    python main.py --once          # 한 번만 실행
"""

import argparse
import logging
import sys

from src.trader import IchimokuTrader

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Ichimoku Cloud 자동매매 봇")
    parser.add_argument(
        "--paper", action="store_true",
        help="페이퍼 모드 (실제 주문 안 보냄)"
    )
    parser.add_argument(
        "--testnet", action="store_true",
        help="Bybit 테스트넷 사용"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="한 번만 실행 (디버깅용)"
    )
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("Ichimoku Cloud 자동매매 봇")
    logger.info("=" * 50)

    mode = "PAPER" if args.paper else "LIVE"
    net = "TESTNET" if args.testnet else "MAINNET"
    logger.info(f"모드: {mode}, 네트워크: {net}")

    try:
        trader = IchimokuTrader(paper=args.paper, testnet=args.testnet)

        if args.once:
            trader.run_once()
        else:
            trader.run()

    except KeyboardInterrupt:
        logger.info("사용자 인터럽트로 종료")
    except Exception as e:
        logger.error(f"치명적 오류: {e}")
        raise


if __name__ == "__main__":
    main()
