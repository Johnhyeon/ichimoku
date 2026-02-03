# 🚀 급등주 전략 실전 배포 가이드

## 📋 포함된 파일 (AI 제외)

### 핵심 전략
- `src/early_surge_detector.py` - 급등 감지 로직
- `src/surge_strategy.py` - 전략 메인
- `src/surge_trader.py` - 트레이더 실행
- `src/bybit_client.py` - Bybit API 클라이언트
- `src/data_fetcher.py` - 실시간 데이터

### 실행 스크립트
- `live_surge.py` - 실시간 트레이딩
- `main.py` - 메인 진입점

### 설정
- `config/.env.example` - 환경변수 예시
- `requirements.txt` - 파이썬 패키지

---

## 🔧 라즈베리파이 설정

### 1. 저장소 클론

```bash
git clone https://github.com/yourusername/ichimoku.git
cd ichimoku
```

### 2. 환경변수 설정

```bash
cp config/.env.example config/.env
nano config/.env
```

**수정 항목:**
```bash
BYBIT_API_KEY=your_real_api_key
BYBIT_API_SECRET=your_real_api_secret
TRADING_ENABLED=true  # 실전 모드
POSITION_SIZE_USDT=100
MAX_POSITIONS=3
```

### 3. 가상환경 및 패키지 설치

```bash
# Python 3.9+ 필요
python3 -m venv venv
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 4. 전략 파라미터 확인

`src/early_surge_detector.py` 열어서 파라미터 확인:

```python
EARLY_SURGE_PARAMS = {
    "volume_spike_min": 10,       # 거래량 10배
    "price_surge_min": 5.0,       # 가격 5% 이상
    "leverage": 5,                # 레버리지 5배
    "sl_pct": 5.0,               # 손절 5%
    "tp_pct": 50.0,              # 익절 50%
}
```

---

## ▶️ 실행

### 실시간 트레이딩

```bash
python live_surge.py
```

### 테스트 모드 (주문 없이)

```bash
# config/.env에서 TRADING_ENABLED=false로 설정
python live_surge.py
```

---

## 📊 전략 요약

### 진입 조건
- ✅ 거래량: 평균 대비 **10배 이상**
- ✅ 가격: **5% 이상** 급등
- ✅ 녹색 캔들
- ✅ 이전 횡보 (변동폭 5% 이하)

### 청산 조건
- 🛑 손절: **-5%**
- 🎯 익절: **+50%**
- 📈 트레일링: 25%부터 시작, 8% 여유

### 리스크 관리
- 레버리지: **5배**
- 포지션 크기: **자산의 3%**
- 최대 동시 포지션: **3개**

---

## 🔒 보안 주의사항

### 절대 Git에 올리지 마세요!
- ❌ `config/.env` (API 키 포함)
- ❌ `data/historical/` (대용량)
- ❌ `models/*.pkl` (ML 모델)

### .gitignore 확인
```bash
git status
# config/.env가 안 보이면 OK!
```

---

## 🐛 문제 해결

### API 연결 에러
```bash
# config/.env 확인
cat config/.env

# API 키 테스트
python -c "from src.bybit_client import BybitClient; c = BybitClient(); print(c.exchange.fetch_balance())"
```

### 패키지 설치 에러
```bash
# 시스템 패키지 업데이트 (라즈베리파이)
sudo apt-get update
sudo apt-get install python3-dev python3-pip

# 재설치
pip install --upgrade pip
pip install -r requirements.txt
```

### 메모리 부족 (라즈베리파이)
```bash
# 스왑 메모리 늘리기
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# CONF_SWAPSIZE=1024 (1GB)
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## 📈 모니터링

### 로그 확인
```bash
# 실시간 로그
tail -f logs/trading.log

# 에러만
tail -f logs/trading.log | grep ERROR
```

### systemd 서비스 등록 (자동 시작)

```bash
# /etc/systemd/system/surge-trading.service
[Unit]
Description=Surge Trading Bot
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/ichimoku
ExecStart=/home/pi/ichimoku/venv/bin/python live_surge.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable surge-trading
sudo systemctl start surge-trading
sudo systemctl status surge-trading
```

---

## ⚠️ 실전 주의사항

### 1. 테스트 먼저!
```bash
# config/.env
TRADING_ENABLED=false  # 테스트 모드
```

### 2. 소액으로 시작
```bash
POSITION_SIZE_USDT=10  # $10부터
```

### 3. 레버리지 낮게
```python
# src/early_surge_detector.py
"leverage": 3,  # 5 → 3으로
```

### 4. 손절 빡빡하게
```python
"sl_pct": 3.0,  # 5 → 3으로
```

### 5. 실시간 모니터링
- 처음 24시간은 계속 확인
- 이상 동작 시 즉시 중단

---

## 📞 긴급 중단

```bash
# 프로세스 종료
pkill -f live_surge.py

# 또는
systemctl stop surge-trading

# 모든 포지션 수동 청산
# Bybit 웹사이트에서 직접 청산
```

---

## 📊 백테스트 결과 참고

**AI 전략 (신뢰도 70%) - 90일:**
- 총 거래: 8건
- 승률: 50%
- 수익률: +287% (레버리지)

**기본 전략 - 90일:**
- (백테스트 실행 필요)

---

## 🔄 업데이트

```bash
# 최신 코드 받기
git pull origin master

# 패키지 업데이트
pip install -r requirements.txt --upgrade

# 재시작
systemctl restart surge-trading
```

---

**행운을 빕니다! 🚀**
