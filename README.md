# ROBU Data Server

FastAPI server that serves live Indian stock data via yfinance to the `robu-valuation-next` Next.js app.

## Requirements

- Python 3.10+
- pip

## Quick Start

```bash
cd /path/to/robu-data-server
bash start.sh
```

The server starts on **http://localhost:8000**.

The Next.js app must be running on **http://localhost:3000** (default `next dev` port).

## Running Both Servers Together

Open two terminal tabs:

**Tab 1 — Python data server:**
```bash
cd ~/Documents/Claude/Projects/Robu\ Terminal/robu-data-server
bash start.sh
```

**Tab 2 — Next.js app:**
```bash
cd ~/Documents/Claude/Projects/Robu\ Terminal/robu-valuation-next
npm run dev
```

Then open http://localhost:3000 in your browser.

## Manual Install

```bash
pip install -r requirements.txt --break-system-packages
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check — returns `{"status": "ok"}` |
| GET | `/search?q=TCS` | Search 50+ Indian stocks by symbol or name |
| GET | `/company/{symbol}` | Full company info (price, PE, market cap, etc.) |
| GET | `/financials/{symbol}` | Last 5 years of annual financials in crores |
| GET | `/price/{symbol}` | Quick price endpoint |

All symbol parameters use NSE symbols **without** the `.NS` suffix (e.g. `RELIANCE`, `TCS`, `INFY`).

## Caching

All yfinance responses are cached in-memory for **15 minutes** to avoid rate limits. Restart the server to clear the cache.

## Fallback Behaviour

If the Python data server is unreachable, all Next.js API routes fall back to the built-in mock data automatically. The app remains fully functional without the data server running.

## Supported Stocks

RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, HDFC, WIPRO, BAJFINANCE, BAJAJFINSV, HINDUNILVR, ITC, KOTAKBANK, LT, AXISBANK, ASIANPAINT, MARUTI, TITAN, ULTRACEMCO, NESTLEIND, SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, TECHM, HCLTECH, POWERGRID, NTPC, ONGC, COALINDIA, SBILIFE, HDFCLIFE, KAYNES, TATAPOWER, TATASTEEL, TATAMOTORS, ADANIENT, ADANIPORTS, JSWSTEEL, HINDALCO, VEDL, BHARTIARTL, BPCL, IOC, GRASIM, EICHERMOT, HEROMOTOCO, BAJAJ-AUTO, M&M, INDUSINDBK, SBIN, LTIM, MPHASIS, PERSISTENT, TATACONSUM, BRITANNIA, DABUR, APOLLOHOSP, FORTIS
