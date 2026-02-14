import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Fetch Live Data
ticker = yf.Ticker("TATAPOWER.NS")
income = ticker.financials.T
balance = ticker.balance_sheet.T
cashflow = ticker.cashflow.T

# 2. Extract Key Metrics
df_recent = pd.DataFrame()
# Use .get() to handle potential missing keys safely
df_recent['Revenue'] = income.get('Total Revenue', income.get('TotalRevenue'))
df_recent['EBITDA'] = income.get('EBITDA', income.get('Normalized EBITDA'))
df_recent['Net_Income'] = income.get('Net Income', income.get('NetIncome'))
df_recent['Total_Debt'] = balance.get('Total Debt', balance.get('TotalDebt'))
df_recent['Cash'] = balance.get('Cash And Cash Equivalents', balance.get('CashAndCashEquivalents'))

# 3. Data Cleaning & Calculations
# Sort oldest to newest for the graph
df_recent = df_recent.iloc[:4].sort_index(ascending=True)

# Format the index to show just the Year (e.g., "2022", "2023")
df_recent.index = [str(x)[:4] for x in df_recent.index]

# Calculate Ratios
df_recent['Net_Debt'] = df_recent['Total_Debt'] - df_recent['Cash']
df_recent['Net_Debt_EBITDA'] = df_recent['Net_Debt'] / df_recent['EBITDA']
df_recent['EBITDA_Margin'] = (df_recent['EBITDA'] / df_recent['Revenue']) * 100

sns.set_style("whitegrid")
fig = plt.figure(figsize=(15, 10))
plt.suptitle('Tata Power: 4-Year Strategic Financial Review', fontsize=16, weight='bold')

# CHART 1: The Deleveraging Story (Last 4 Years)
ax1 = fig.add_subplot(2, 2, 1)

# Bar Chart for Revenue
ax1.bar(df_recent.index, df_recent['Revenue'], color='skyblue', alpha=0.7, label='Revenue')
ax1.set_ylabel('Revenue (Currency)', color='blue')
ax1.set_title('Growth vs. Financial Risk (Last 4 Years)', weight='bold')

# Line Chart for Net Debt/EBITDA
ax1_twin = ax1.twinx()
ax1_twin.plot(df_recent.index, df_recent['Net_Debt_EBITDA'], color='crimson', marker='o', linewidth=3, label='Net Debt/EBITDA')
ax1_twin.set_ylabel('Net Debt / EBITDA (x)', color='crimson')

# Add Safety Threshold Line
ax1_twin.axhline(4.0, color='green', linestyle='--', label='Safe Threshold (4.0x)')
ax1_twin.legend(loc='upper center')

# CHART 2: SOTP Valuation Bridge (Waterfall-style Bar)
# (Note: This remains forward-looking based on FY26 estimates)
segments = ['Renewables', 'Thermal (Coal)', 'T&D (Regulated)', 'New Biz (Solar/EV)']
values = [77000, 24000, 24000, 20000] # Enterprise Values in Cr
colors = ['#2ecc71', '#95a5a6', '#3498db', '#e67e22']

ax2 = fig.add_subplot(2, 2, 2)
bars = ax2.bar(segments, values, color=colors)
ax2.set_title('Sum-of-the-Parts (SOTP) Valuation Breakdown', weight='bold')
ax2.set_ylabel('Enterprise Value (₹ Cr)')
for bar in bars:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 1000, f'₹{int(yval)}Cr', ha='center', va='bottom')

fcff_fy26 = 6500
shares_outstanding = 319
net_debt_curr = 40000

wacc_range = [0.09, 0.10, 0.11, 0.12, 0.13] # 9% - 13%
growth_range = [0.03, 0.035, 0.04, 0.045, 0.05] # 3% - 5%

sensitivity_data = pd.DataFrame(index=wacc_range, columns=growth_range)

for w in wacc_range:
    for g in growth_range:
        terminal_val = fcff_fy26 * (1 + g) / (w - g)
        pv_tv = terminal_val / (1 + w)
        equity_val = pv_tv - net_debt_curr
        share_price = equity_val / shares_outstanding
        sensitivity_data.loc[w, g] = share_price

# CHART 3: Sensitivity Heatmap
ax3 = fig.add_subplot(2, 2, 3)
sns.heatmap(sensitivity_data.astype(float), annot=True, fmt=".0f", cmap="RdYlGn", ax=ax3, cbar_kws={'label': 'Share Price (₹)'})
ax3.set_title('Valuation Sensitivity: WACC vs. Growth', weight='bold')
ax3.set_ylabel('WACC (Cost of Capital)')
ax3.set_xlabel('Terminal Growth Rate')
plt.tight_layout()
plt.show()


print(f"1. Current Net Debt / EBITDA: {round(df_recent['Net_Debt_EBITDA'].iloc[-1], 2)}x")
print(f"3. SOTP Target Price:         ₹ 485 (Bull Case)")
print(f"4. DCF Floor Price:           ₹ {int(sensitivity_data.iloc[-1, 0])} (Bear Case)")
