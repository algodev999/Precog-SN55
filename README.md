# 🧭 CoinMetrics Precog Subnet (SN-55)

A Bittensor subnet that identifies the best analysts and strategies to anticipate Bitcoin price movements—rewarding precise point and interval forecasts with high-frequency (5‑min) predictions and hourly resolution.

---

## 📚 Table of Contents

[About](#-about)  
[Features](#-features)  
[Tech Stack](#-tech-stack)  
[Installation](#️-installation)  
[Usage](#-usage)  
[Configuration](#-configuration)  
[Screenshots](#-screenshots)  
[API Documentation](#-api-documentation)  
[Contact](#-contact)  
[Acknowledgements](#-acknowledgements)

---

## 🧩 About

This project provides a decentralized arena for Bitcoin price prediction on the Bittensor network. It rewards miners for two types of forecasts: a **point estimate** (BTC price in USD one hour ahead) and an **interval estimate** (min–max price band over the next hour at 1s frequency). The incentive mechanism is designed to surface the most accurate and useful signals, with high-frequency updates that traditional derivatives cannot offer.

**Key goals:** Focus on Bitcoin for data quality and volatility; 5‑minute prediction cadence with 1‑hour resolution; precise level and band forecasts instead of coarse long/short or fixed strikes.

---

## ✨ Features

- **Point & interval forecasts** – Miners submit both a 1‑hour-ahead price level and a min–max band for the next hour.
- **Custom miner strategies** – Plug in your own forward function (e.g. GARCH, ensemble, empirical distributions) via `FORWARD_FUNCTION`.
- **Validator auto-update** – Optional PM2-based auto-updater keeps validators on the latest code.
- **WandB integration** – Validators can log runs to Weights & Biases for monitoring and analysis.
- **Multiple network modes** – Run on localnet, testnet (UID 256), or mainnet Finney (UID 55).

---

## 🧠 Tech Stack

| Category   | Technologies |
| ---------- | ------------ |
| **Languages** | Python 3.9–3.11 |
| **Frameworks** | Bittensor 9.x, Pydantic |
| **Data** | CoinMetrics API, pandas, numpy |
| **Tools** | Poetry, PM2, pre-commit, Black, pytest |

---

## ⚙️ Installation

```bash
# Install PM2 (for process management)
sudo apt update
sudo apt install nodejs npm
sudo npm install pm2@latest -g
```

```bash
# Clone the repository
git clone https://github.com/coinmetrics/precog.git
cd precog
```

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

```bash
# Install dependencies with Poetry
pip install poetry
poetry install
```

---

## 🚀 Usage

**Register a hotkey (once):**

```bash
make register ENV_FILE=.env.miner    # or .env.validator
```

**Run a miner (base or custom):**

```bash
# Base miner
make miner ENV_FILE=.env.miner

# Custom miner (set FORWARD_FUNCTION in .env.miner)
make miner ENV_FILE=.env.miner
```

**Run a validator:**

```bash
make validator ENV_FILE=.env.validator
```

For a **custom miner**, add a module under `precog/miners/` that defines a `forward` function (setting `synapse.predictions` and `synapse.interval`), then set `FORWARD_FUNCTION=your_module_name` in `.env.miner`.

---

## 🧾 Configuration

Copy the appropriate env template and edit values.

**Validator:** `cp .env.validator.example .env.validator`  
**Miner:** `cp .env.miner.example .env.miner`

**Example `.env.miner`:**

```env
NETWORK=finney
COLDKEY=your_miner_coldkey
MINER_HOTKEY=your_miner_hotkey
MINER_NAME=miner
MINER_PORT=8092
TIMEOUT=16
VPERMIT_TAO_LIMIT=2
FORWARD_FUNCTION=base_miner
LOGGING_LEVEL=debug
LOCALNET=ws://127.0.0.1:9945
```

**Example `.env.validator` (validators only):**

- Set `WANDB_API_KEY` and run `wandb login` if using Weights & Biases.
- Set `AUTO_UPDATE=1` to enable automatic code updates via the bundled script.
 
---

## 📜 API Documentation

This project runs as a **Bittensor subnet** (miners and validators). There is no REST API; communication follows the subnet protocol.

- **Protocol & rewards:** [Coin Metrics – Precog Methodology](https://docs.coinmetrics.io/bittensor/precog-methodology)
- **Reward logic:** `precog/validators/reward.py`
- **Base miner contract:** `precog/miners/miner.py` and custom forward modules in `precog/miners/`

---

## 📬 Contact

- **Subnet:** CoinMetrics Precog — Testnet UID: 256 · Mainnet UID: 55  
- **Authors:** PJS
- **Emails:** cpalvarez95999@gmail.com 
- **Resources:** [Bittensor Docs](https://docs.bittensor.com/) · [CoinMetrics](https://charts.coinmetrics.io/crypto-data/)

---

## 🌟 Acknowledgements

- [Bittensor](https://bittensor.com/) for the decentralized ML network.
- [CoinMetrics](https://coinmetrics.io/) for crypto data and API.
- [Weights & Biases](https://wandb.ai/) for experiment tracking (validators).
- Inspiration from the original [CoinMetrics Precog](https://github.com/coinmetrics/precog) repository.
