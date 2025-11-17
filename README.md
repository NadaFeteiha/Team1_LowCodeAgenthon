## Project Overview
**Project Name:** Supply Chain - Warehousing  
**Agent Name:** Inventory Management Agent / Medical Inventory Management Agent  
**Team:** Team 1 - Supply Soul  
**Contributors:** Megha Narendra Simha, Poorrnima Vetrivelan, Nada Feteiha

This system helps healthcare facilities efficiently manage medication and supply inventory, forecast demand using **XGBoost**, and automate restocking. It leverages AI agents, a Flask backend, PostgreSQL database, and a React frontend with visualizations.

---

## Key Goals
- Maintain real-time inventory levels and generate alerts for low stock.
- Analyze historical consumption data and forecast future demand.
- Automate purchase orders with human-in-the-loop approval.
- Visualize inventory, demand, and forecasts in an interactive dashboard.

---

## Architecture Overview
- **Frontend (React):** Dashboard with **React Charts** for stock and forecast visualization, human approval UI for purchase orders.
- **Backend (Python - Flask / FastAPI):** MCP orchestrator, API endpoints, semantic search using embeddings, orchestration of Stock, Demand, and Reorder agents.
- **AI Agents:**
  1. **Stock Level Monitor:** Tracks inventory, generates low-stock alerts, summarizes inventory in CSV and charts.
  2. **Demand Forecaster:** Uses **XGBoost** for demand prediction; analyzes historical trends and seasonal patterns.
  3. **Reorder Automator:** Generates purchase orders based on low-stock and forecast, supports human approval, updates inventory and order tracking.
- **Database (PostgreSQL):** Stores historical consumption, stock, vendor data; tables include `inventory_master`, `consumption`, `finance`, `vendor_master`, `inventory_department_mapping`.
## Medical Inventory Management System

**Project Name:** Supply Chain - Warehousing

**Agent Name:** Inventory Management Agent / Medical Inventory Management Agent

**Team:** Team 1 - Supply Soul

**Contributors:** Megha Narendra Simha, Poorrnima Vetrivelan, Nada Feteiha

Overview
--------
This repository implements an inventory management system for healthcare facilities. It provides:
- real-time stock tracking and low-stock alerts
- demand forecasting using XGBoost
- automated purchase-order generation with human-in-the-loop approval
- a React dashboard for visualization

Architecture
------------
- Frontend: React (in `Frontend/`) — dashboard, charts, approval UI
- Backend: Python Flask (in `Backend/`) — API, MCP orchestration, semantic search
- AI: XGBoost demand forecaster, sentence-transformer embeddings for semantic search
- Database: PostgreSQL (schema files at repo root)

Requirements
------------
- macOS / Linux / Windows
- Python 3.9+
- Node.js + npm or yarn
- PostgreSQL
- `ngrok` (optional, for public tunneling)

Consolidated Setup (single block)
---------------------------------
Run the commands below from your shell (zsh on macOS). Replace placeholders like `<repo_url>`, `<db_user>`, and `<db_name>`.

```bash
# 1. Clone repository
git clone <repo_url>
cd Team1_LowCodeAgenthon

# 2. Backend: create & activate virtualenv (macOS/Linux)
cd Backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install fastmcp

# 3. PostgreSQL: load schema and (optional) full dump (run from repo root)
# Make sure PostgreSQL is running and you have a user & database created
psql -U <db_user> -d <db_name> -f ../schema_only.sql
psql -U <db_user> -d <db_name> -f ../full_dump.sql

# 4. Train XGBoost model (optional) or use pre-trained model files
# Notebook included at Backend/xgboostmodel/demand_forecast.ipynb
jupyter notebook ../Backend/xgboostmodel/demand_forecast.ipynb

# 5. Generate sentence-transformer embeddings for semantic search
python3 semantic_search/vectorembedding.py

# 6. Start MCP orchestration (if applicable)
python3 semantic_search/combine_mcp_demand_stock_withss.py

# 7. (Optional) expose local backend with ngrok
# install ngrok separately and run:
ngrok http 8000

# 8. Start Flask backend (adjust port or env vars as needed)
python3 app.py

# 9. Frontend: open a new terminal, install and run dev server
cd ../Frontend
npm install
npm run dev

# 10. Visit the app in your browser (default Vite port is 5173)
echo "Frontend running at http://localhost:5173"

```

Notes
-----
- File locations in this repo:
  - Backend code: `Backend/` (contains `app.py`, API modules, `semantic_search/`)
  - Frontend app: `Frontend/` (Vite + React)
  - Model notebooks: `Backend/xgboostmodel/`
  - DB schema & dumps: `schema_only.sql`, `full_dump.sql` (repo root)

- If you run into missing package errors, activate the virtualenv (`source Backend/venv/bin/activate`) and install the missing package with `pip install <pkg>`.
- Use Python 3.9+ and `python3` explicitly on macOS to avoid conflicts with system Python.

Development tips
----------------
- To run backend APIs locally: activate the Backend virtualenv and run `python3 app.py`.
- To retrain the demand model, open the Jupyter notebook in `Backend/xgboostmodel/`.
- For semantic search troubleshooting, check the embeddings file (if created) and the `semantic_search/` scripts.
- Test queries in Test_queries.docx

Future enhancements
-------------------
- Voice interface (React + Web Speech API)
- Email notifications for low-stock alerts and POs


