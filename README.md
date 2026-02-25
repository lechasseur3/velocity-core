# Velocity Core

![Velocity Core Dashboard](https://raw.githubusercontent.com/lechasseur3/velocity-core/main/screenshot.png)

Velocity Core is an open-source enterprise-grade quantitative portfolio optimization engine. It utilizes the Fama-French 5-Factor model, Black-Litterman optimization, and Efficient Frontier calculations to help you manage your investments with a data-driven approach.

Crafted by **Karl BAUJON**.

## Features

- **Fama-French 5-Factor Analysis**: Automatically calculate expected returns based on academic factor exposures (Market Risk, Small Caps, Value Stocks, Profitability, Conservative Investment).
- **Black-Litterman Optimization**: Combine market equilibrium with your personal absolute or relative market convictions.
- **Efficient Frontier (Markowitz)**: Find the optimal asset allocation that maximizes the Sharpe ratio.
- **Risk Metrics**: Calculate Value at Risk (VaR), Max Drawdown, Beta, Alpha, and Portfolio Volatility.
- **Dynamic React Dashboard**: Interactive analytics and visualizations built with a stunning, premium dark-mode interface.

## Prerequisites

- Node.js (v18 or higher)
- Python 3.9+
- pip (Python package manager)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/lechasseur3/velocity-core.git
cd velocity-core
```

### 2. Install Backend Dependencies (Python)

The backend relies on several quantitative and financial libraries.
It is highly recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

*Required packages include: `numpy`, `pandas`, `yfinance`, `scipy`, `pypfopt`, `plotly`, `pandas-datareader`, `statsmodels`, `rich`.*
*Note: `pandas-datareader` and `statsmodels` are essential for the Fama-French 5-Factor analysis.*

### 3. Install Frontend Dependencies (React)

Open a new terminal window/tab:

```bash
cd frontend
npm install
```

## Usage

### Running the Full Web Platform (React + FastAPI)

The easiest way to run the entire application (both the Python backend and the React frontend) is to use the provided bash script.

```bash
chmod +x dev.sh
./dev.sh
```
This will automatically:
1. Start the FastAPI backend on `http://localhost:8000`
2. Start the Vite React development server on `http://localhost:5173`

*To stop the servers, simply press `Ctrl+C` in the terminal.*

### Running the CLI Engine & Plotly Dashboard

If you prefer to run the quantitative engine purely in the terminal and view the complete Plotly Executive Dashboard:

```bash
python Portfolio_Velocity1_Karl.py
```
*Note: You may need to edit the `SYMBOLS` list inside `Portfolio_Velocity1_Karl.py` to change the default analyzed assets.*

## Legal Disclaimer

**WARNING:**
This software is provided for EDUCATIONAL purposes only. It does NOT constitute financial or investment advice. Past performance is not indicative of future results. Investing in the stock market involves a high degree of risk, including the risk of losing your entire invested capital.

## License

This project is licensed under a **Custom Non-Commercial License**. 

You are free to use, study, and modify this software for personal and educational purposes. However, **any commercial use, monetization, or integration into a profitable service is strictly prohibited without explicit written consent from the author, Karl BAUJON.** 

For full terms and conditions, please see the [LICENSE](LICENSE) file in this repository.
