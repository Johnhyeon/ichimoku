# Ichimoku Cloud Trading Bot

백테스트 검증 완료된 일목균형표 자동매매 봇입니다.

## 백테스트 성과 (2024-04 ~ 2026-01)

- **총 수익**: $142,257 (6,777% 수익률)
- **승률**: 41.9%
- **Profit Factor**: 1.73
- **LONG 수익**: +$1,359 (31.0% 승률)
- **SHORT 수익**: +$140,899 (44.9% 승률)
- **MDD**: 42.2% (레버리지 20배 기준)

## 핵심 전략

### 1. BTC 도미넌스 필터 (Strict Mode)
- **BTC 상승 추세** (MA26 > MA52): SHORT만 진입
- **BTC 하락 추세** (MA26 < MA52): LONG만 진입

### 2. LONG Volume Spike 필터 (Filter4)
- 후행스팬 상승 **필수**
- 거래량 > 평균의 120% **필수**
- 이 필터로 LONG이 손실에서 수익으로 전환 (-$8,434 → +$1,359)

### 3. SHORT 전략 (기존 유지)
- 가격 < 구름 하단
- 전환선 < 기준선 + 크로스 신호

## 설치

```bash
pip install -r requirements.txt
```

## 설정

1. `config/.env.example`을 복사하여 `config/.env` 생성
2. Bybit API 키 입력

```bash
cp config/.env.example config/.env
# .env 파일 편집하여 API 키 입력
```

## 실행

### 1. 신호 확인 (안전)
```bash
python ichimoku_live.py --testnet --once
```

### 2. 페이퍼 트레이딩
```bash
python ichimoku_live.py --paper
```

### 3. 테스트넷 실전
```bash
python ichimoku_live.py --testnet
```

### 4. 메인넷 실전 (주의!)
```bash
python ichimoku_live.py
```

## 권장 리스크 관리

실전 운용 시:

1. **레버리지**: 10배 권장 (20배 → 10배 시 MDD 50% 감소)
2. **포지션 크기**: 2-3% 권장 (기본 5%)
3. **초기 자본**: 전체의 10-20%로 시작
4. **일일 손실 한도**: 전체 자본의 5%

## 전략 파라미터

```python
LEVERAGE = 20  # 레버리지 (10배 권장)
POSITION_PCT = 0.05  # 포지션 크기 (5%)

STRATEGY_PARAMS = {
    "min_cloud_thickness": 0.2,
    "min_sl_pct": 0.3,
    "max_sl_pct": 8.0,
    "sl_buffer": 0.2,
    "rr_ratio": 2.0,
    "trail_pct": 1.5,
    "cooldown_hours": 4,
    "max_positions": 5,
    "use_btc_filter": True,  # BTC 도미넌스 필터
    "long_chikou_required": True,  # LONG 후행스팬 필수
    "long_volume_min_ratio": 1.2,  # LONG 거래량 필터
}
```

## 거래 대상 코인 (20개)

BTC, ETH, BNB, XRP, SOL, ADA, DOGE, AVAX, DOT, LINK,
MATIC, LTC, ATOM, UNI, ETC, APT, NEAR, FIL, AAVE, INJ

## 주의사항

⚠️ **이 봇은 실제 자금을 사용합니다**
- 먼저 테스트넷에서 충분히 테스트하세요
- 소액으로 시작하여 점진적으로 확대하세요
- 시장 상황에 따라 손실이 발생할 수 있습니다
- 투자 손실에 대한 책임은 사용자에게 있습니다

## 라이선스

MIT License
