#!/usr/bin/env python3
"""
Early Surge 전략 실시간 트레이더 실행 스크립트

백테스트 결과:
  - 총 수익률: +197.4% (13거래, 승률 38.5%)
  - 평균 이익: +97.2% vs 평균 손실: -22.1%
  - 리스크/리워드: 4.4:1

실행 예시:
    # 페이퍼 모드 (시뮬레이션, 실제 주문 안 나감)
    python live_surge.py --paper

    # 실제 거래 (메인넷, 신중히!)
    python live_surge.py

    # 테스트넷 (가상 자금)
    python live_surge.py --testnet

주의사항:
    ⚠️  실제 거래는 큰 손실 위험이 있습니다!
    ⚠️  소액으로 시작하고 반드시 손실 한도를 설정하세요!
    ⚠️  암호화폐 시장은 변동성이 매우 크므로 투자에 주의하세요!
"""

import argparse
import logging
import sys

from src.surge_trader import SurgeTrader

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("surge_bot.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Early Surge 전략 실시간 트레이더",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  페이퍼 모드 (시뮬레이션):
    python live_surge.py --paper --initial 1000

  테스트넷 (가상 자금):
    python live_surge.py --testnet --initial 500

  실제 거래 (메인넷, 신중히!):
    python live_surge.py --initial 500 --loss-limit 20 --max-positions 3

⚠️  주의: 암호화폐 거래는 큰 손실 위험이 있습니다!
        """
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="페이퍼 모드 (실제 주문 안 보냄, 시뮬레이션만)"
    )
    parser.add_argument(
        "--testnet", action="store_true",
        help="Bybit 테스트넷 사용 (가상 자금)"
    )
    parser.add_argument(
        "--initial", type=float, default=1000.0,
        help="초기 운용 자금 (기본: 1000 USDT)"
    )
    parser.add_argument(
        "--loss-limit", type=float, default=20.0,
        help="일일 손실 한도 %% (초기 자금 대비, 기본: 20%%)"
    )
    parser.add_argument(
        "--max-positions", type=int, default=3,
        help="최대 동시 포지션 수 (기본: 3개)"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="한 번만 실행 (디버깅용)"
    )

    args = parser.parse_args()

    # 안전 확인
    if not args.paper and not args.testnet:
        print("\n" + "="*70)
        print("⚠️  실제 거래 모드입니다!")
        print("="*70)
        print(f"초기 자금: ${args.initial:,.0f}")
        print(f"일일 손실 한도: ${args.initial * args.loss_limit / 100:,.0f} ({args.loss_limit}%)")
        print(f"최대 포지션: {args.max_positions}개")
        print()
        print("암호화폐 거래는 큰 손실 위험이 있습니다!")
        print("소액으로 시작하고 감당할 수 있는 금액만 투자하세요!")
        print()
        confirm = input("계속하시겠습니까? (yes 입력): ")
        if confirm.lower() != "yes":
            print("취소되었습니다.")
            return

    logger.info("=" * 70)
    logger.info("Early Surge 전략 실시간 트레이더")
    logger.info("=" * 70)

    mode = "PAPER" if args.paper else "LIVE"
    net = "TESTNET" if args.testnet else "MAINNET"
    logger.info(f"모드: {mode}, 네트워크: {net}")
    logger.info(f"초기 자금: ${args.initial:,.0f}")
    logger.info(f"일일 손실 한도: ${args.initial * args.loss_limit / 100:,.0f} ({args.loss_limit}%)")
    logger.info(f"최대 포지션: {args.max_positions}개")

    try:
        trader = SurgeTrader(
            paper=args.paper,
            testnet=args.testnet,
            initial_balance=args.initial,
            daily_loss_limit_pct=args.loss_limit,
            max_positions=args.max_positions
        )

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
