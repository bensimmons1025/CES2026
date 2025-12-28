# =========================================================
# [Final Complete Ver] AI Strategy vs QQQ
# =========================================================
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.gridspec as gridspec
import random

# 시드 고정(재현성 확보)
TARGET_SEED = 9

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed fixed to {seed}.")

# --- 모델 아키텍처 ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    def forward(self, Q, K, V):
        Q, K, V = self.W_Q(Q), self.W_K(K), self.W_V(V)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        weights = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.depth)
        weights = torch.softmax(weights, dim=-1)
        return self.W_O(self._combine_heads(torch.matmul(weights, V)))
    def _split_heads(self, x): return x.view(x.size(0), -1, self.num_heads, self.depth).transpose(1, 2)
    def _combine_heads(self, x): return x.transpose(1, 2).contiguous().view(x.size(0), -1, self.num_heads * self.depth)

class Transformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(Transformer, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                MultiHeadAttention(hidden_dim, num_heads),
                nn.LayerNorm(hidden_dim),
                nn.Sequential(nn.Linear(hidden_dim, 4*hidden_dim), nn.ReLU(), nn.Linear(4*hidden_dim, hidden_dim)),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.encoder_layers:
            attn = layer[0](x, x, x)
            x = layer[1](x + attn)
            ff = layer[2](x)
            x = layer[3](x + ff)
        return self.output_layer(x[:, -1, :])

# --- 시뮬레이션 함수 ---
def run_simulation_final(seed_val):
    set_seed(seed_val)

    print("📥 데이터 다운로드 및 학습 시작...")
    stocks = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'AVGO', 'TSLA', 'GOOG', 'COST', 'NFLX', 'PLTR']
    macros = ['^TNX', '^VIX', 'DX-Y.NYB']
    tickers = macros + stocks

    end_date = datetime.today()
    start_date = end_date - timedelta(days=365*8)

    data_raw = yf.download(tickers, start=start_date, end=end_date, progress=False)
    closes = data_raw['Close'].fillna(method='ffill').fillna(method='bfill')
    qqq = yf.download('QQQ', start=start_date, end=end_date, progress=False)['Close']

    test_start = end_date - timedelta(days=365*5)
    train_data = closes.loc[:test_start]

    train_norm = (train_data - train_data.min()) / (train_data.max() - train_data.min())
    inputs, labels = [], []
    values = train_norm.values
    for i in range(len(values) - 31):
        inputs.append(values[i:i+30])
        ret = (values[i+30, 3:] - values[i+29, 3:]) / values[i+29, 3:]
        ret = np.nan_to_num(ret)
        rank = np.argsort(np.argsort(ret))
        label = (rank >= (len(stocks) - 3)).astype(float)
        labels.append(label)

    inputs = torch.tensor(np.array(inputs), dtype=torch.float32)
    labels = torch.tensor(np.array(labels), dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(len(tickers), 128, 4, 2, len(stocks)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 10 Epoch
    print("🤖 모델 학습 중 (Epoch 10)...", end="")
    for epoch in range(10):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    print(" 완료!")

    # --- 백테스트 ---
    initial_capital = 10000.0
    capital = initial_capital
    history = []
    dates = []
    selection_history = []

    current_date = test_start
    qqq_ma200 = qqq.rolling(window=200).mean()

    # QQQ Benchmark
    qqq_bench = qqq.loc[test_start:end_date]
    try: base_price = qqq_bench.iloc[0].item()
    except: base_price = qqq_bench.iloc[0]
    qqq_norm = qqq_bench / base_price * 100

    while current_date < end_date:
        next_date = current_date + relativedelta(months=1)
        if next_date > end_date: next_date = end_date
        str_curr = current_date.strftime('%Y-%m-%d')
        str_next = next_date.strftime('%Y-%m-%d')

        # 1. Market Filter
        try:
            curr_qqq = qqq.loc[str_curr]
            curr_ma200 = qqq_ma200.loc[str_curr]
            if isinstance(curr_qqq, pd.Series): curr_qqq = curr_qqq.iloc[0]
            if isinstance(curr_ma200, pd.Series): curr_ma200 = curr_ma200.iloc[0]
            is_bull_market = curr_qqq > curr_ma200
        except: is_bull_market = True

        if not is_bull_market:
            history.append(capital)
            dates.append(next_date)
            selection_history.append([next_date, "CASH", "CASH", "CASH"])
            current_date = next_date
            continue

        # 2. AI Selection
        try:
            train_slice = closes.loc[:str_curr]
            if len(train_slice) < 60:
                current_date = next_date
                continue
            window = train_slice.iloc[-60:]
            norm_val = (window - window.min()) / (window.max() - window.min())
            input_tensor = torch.tensor(norm_val.values[-30:], dtype=torch.float32).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad(): scores = model(input_tensor).cpu().numpy()[0]
            top_picks = pd.Series(scores, index=stocks).sort_values(ascending=False).head(3).index.tolist()

            selection_history.append([next_date] + top_picks)

            period_prices = closes.loc[str_curr:str_next, top_picks]
            if len(period_prices) > 0:
                period_return = (period_prices.iloc[-1] - period_prices.iloc[0]) / period_prices.iloc[0]
                avg_return = period_return.mean()
                capital = capital * (1 + avg_return)
        except:
            pass

        history.append(capital)
        dates.append(next_date)
        current_date = next_date

    return history, dates, selection_history, stocks, qqq_norm

# --- 시각화 함수 (에러 수정됨) ---
def plot_final_results(history, dates, selection_history, stocks, qqq_norm):
    # AI 데이터 정리
    strategy_curve = pd.Series(history, index=dates)
    initial_val = history[0]
    final_val = history[-1]

    # QQQ 데이터 정리 (10,000달러 시작으로 스케일링)
    qqq_scaled = qqq_norm / 100 * initial_val

    # .item()으로 스칼라 값 강제 추출
    try:
        qqq_final_val = qqq_scaled.iloc[-1].item()
    except:
        qqq_final_val = qqq_scaled.iloc[-1]

    # 수익률(%) 계산
    ai_return_pct = (final_val - initial_val) / initial_val * 100
    qqq_return_pct = (qqq_final_val - initial_val) / initial_val * 100

    # 시각화 준비
    sel_df = pd.DataFrame(selection_history, columns=['Date', 'Pick1', 'Pick2', 'Pick3'])

    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.2, 1])

    # (1) 수익률 비교 그래프
    ax1 = plt.subplot(gs[0])

    ax1.plot(strategy_curve.index, strategy_curve.values,
             color='purple', linewidth=2.5,
             label=f'AI Strategy (Return: +{ai_return_pct:.2f}%)')

    ax1.plot(qqq_scaled.index, qqq_scaled.values,
             color='orange', linewidth=2, linestyle='--', alpha=0.8,
             label=f'QQQ Benchmark (Return: +{qqq_return_pct:.2f}%)')

    ax1.set_title(f'Performance Comparison (Seed {TARGET_SEED})\nAI: ${final_val:,.0f} vs QQQ: ${qqq_final_val:,.0f}',
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left', fontsize=12, frameon=True)
    ax1.grid(True, alpha=0.3)

    # Alpha 표시
    diff = ai_return_pct - qqq_return_pct
    status = "Outperforming" if diff > 0 else "Underperforming"
    ax1.text(0.02, 0.85, f"Alpha: {diff:+.2f}%p ({status})", transform=ax1.transAxes,
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # (2) 종목 선택 히트맵
    ax2 = plt.subplot(gs[1], sharex=ax1)
    all_assets = ['CASH'] + stocks

    for date_idx, row in sel_df.iterrows():
        date = row['Date']
        picks = [row['Pick1'], row['Pick2'], row['Pick3']]
        if 'CASH' in picks:
            ax2.axvspan(date - timedelta(days=15), date + timedelta(days=15), color='gray', alpha=0.2)
            ax2.scatter(date, 'CASH', color='gray', s=100, marker='s')
        else:
            for pick in picks:
                if pick in all_assets:
                    if pick in ['NVDA', 'PLTR', 'TSLA']:
                        ax2.scatter(date, pick, color='crimson', s=100, alpha=0.9, edgecolors='white')
                    else:
                        ax2.scatter(date, pick, color='royalblue', s=60, alpha=0.6)

    ax2.set_title('AI Monthly Stock Selection (Red: High Volatility Picks)', fontsize=14, fontweight='bold')
    ax2.set_yticks(range(len(all_assets)))
    ax2.set_yticklabels(all_assets)
    ax2.grid(True, alpha=0.3, axis='x')
    ax2.set_ylabel('Selected Assets')

    plt.tight_layout()
    plt.show()

    print("="*50)
    print(f"📊 [최종 결과] Seed {TARGET_SEED}")
    print(f" - AI 최종 자산: ${final_val:,.2f} (+{ai_return_pct:.2f}%)")
    print(f" - QQQ 최종 자산: ${qqq_final_val:,.2f} (+{qqq_return_pct:.2f}%)")
    print(f"👉 결과: {'AI 승리 🏆' if ai_return_pct > qqq_return_pct else 'QQQ 승리'}")
    print("="*50)

if __name__ == "__main__":
    hist, dts, sel_h, st_list, qqq_n = run_simulation_final(TARGET_SEED)
    plot_final_results(hist, dts, sel_h, st_list, qqq_n)