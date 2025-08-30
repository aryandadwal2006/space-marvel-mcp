# Universal MCP Client - Complete Setup Instructions

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+
- A Google AI Studio API key ([Get one here](https://aistudio.google.com/app/apikey))

### 1. Project Structure
Create the following directory structure:

```
spacemarvel_mcp/
├── app.py                          # Main Streamlit app (provided)
├── core/
│   ├── __init__.py                # Empty file
│   ├── config_loader.py           # Provided
│   ├── mcp_registry.py           # Provided  
│   ├── orchestrator.py           # Provided
│   └── transport_adapters.py     # Provided
├── configs/                       # JSON configs directory
│   ├── default_user/             # User-specific configs
│   │   ├── stripe_payments.json  # Provided
│   │   └── filesystem.json       # Provided
├── schemas/
│   └── server.schema.json        # Provided
├── requirements.txt              # Provided
└── README.md
```

### 2. Installation

1. **Clone or create the project directory:**
```bash
mkdir spacemarvel_mcp
cd spacemarvel_mcp
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Create directory structure:**
```bash
mkdir -p core configs/default_user schemas
touch core/__init__.py
```

### 3. Configuration Files Setup

1. **Copy the provided files to their correct locations:**
   - `server.schema.json` → `schemas/server.schema.json`
   - `stripe_payments.json` → `configs/default_user/stripe_payments.json`
   - `filesystem.json` → `configs/default_user/filesystem.json`
   - All Python files → respective directories

### 4. Environment Setup

Create a `.env` file (optional):
```bash
GEMINI_API_KEY=your_gemini_api_key_here
STRIPE_SECRET_KEY=your_stripe_key_here  # For Stripe server
FILESYSTEM_ROOT=/tmp                    # For filesystem server
```

### 5. Running the Application

```bash
streamlit run app.py
```