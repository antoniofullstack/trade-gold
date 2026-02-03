from collections import deque
import numpy as np
import pandas as pd


class GoldTradingEnv:
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance=10_000,
        trade_size=1.0,
        transaction_cost=0.001,
        stop_loss_pct=0.01,
        take_profit_pct=0.02
    ):
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.trade_size = trade_size
        self.transaction_cost = transaction_cost
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self._prepare_arrays()

        self.reset()

    def _prepare_arrays(self):
        self.data = self.data.reset_index(drop=True)
        self.close_prices = self.data["Close"].to_numpy(dtype=np.float32)
        self.returns = self.data["Return"].to_numpy(dtype=np.float32)
        self.sma_fast = self.data["SMA_fast"].to_numpy(dtype=np.float32)
        self.sma_slow = self.data["SMA_slow"].to_numpy(dtype=np.float32)
        self.rsi = self.data["RSI"].to_numpy(dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0      # 0, 1, -1
        self.entry_price = 0.0
        self.current_step = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance

        # Histórico para visualização
        self.equity_history = []
        self.drawdown_history = []
        self.trade_history = []
        self.returns_window = deque(maxlen=20)

        return self._get_state()

    def step(self, action):
        done = False
        reward = 0.0
        
        # Guardar net worth anterior para calcular retorno
        prev_net_worth = self.net_worth
        
        price = float(self.close_prices[self.current_step])
        trade_value = price * self.trade_size
        fee = trade_value * self.transaction_cost

        # ---- AÇÃO: BUY (1) ----
        if action == 1:
            # abrir long
            if self.position == 0:
                cost = trade_value + fee
                if self.balance >= cost:
                    self.balance -= cost
                    self.position = 1
                    self.entry_price = price
                    self.trade_history.append((self.current_step, price, "BUY"))

            # fechar short
            elif self.position == -1:
                # recomprar para fechar short
                self.balance -= trade_value + fee
                self.position = 0
                self.entry_price = 0.0
                self.trade_history.append((self.current_step, price, "CLOSE_SHORT"))

        # ---- AÇÃO: SELL (2) ----
        elif action == 2:
            # abrir short
            if self.position == 0:
                # vender a descoberto: recebe o valor da venda
                self.balance += trade_value - fee
                self.position = -1
                self.entry_price = price
                self.trade_history.append((self.current_step, price, "SELL"))

            # fechar long
            elif self.position == 1:
                # vender para fechar long
                self.balance += trade_value - fee
                self.position = 0
                self.entry_price = 0.0
                self.trade_history.append((self.current_step, price, "CLOSE_LONG"))

        # ---- AÇÃO: CLOSE (3) ----
        elif action == 3 and self.position != 0:
            if self.position == 1:
                # fechar long
                self.balance += trade_value - fee
            else:
                # fechar short
                self.balance -= trade_value + fee
            self.position = 0
            self.entry_price = 0.0
            self.trade_history.append((self.current_step, price, "CLOSE"))

        # ---- ATUALIZA NET WORTH ----
        # Net worth = caixa + posição a mercado
        self.net_worth = self.balance + (self.position * trade_value)

        # ---- STOP LOSS E TAKE PROFIT AUTOMÁTICOS ----
        if self.position != 0:
            if self.position == 1:  # LONG
                if price <= self.entry_price * (1 - self.stop_loss_pct):
                    self.balance += trade_value - fee
                    self.position = 0
                    self.entry_price = 0.0
                    self.trade_history.append((self.current_step, price, "STOP_LOSS"))
                elif price >= self.entry_price * (1 + self.take_profit_pct):
                    self.balance += trade_value - fee
                    self.position = 0
                    self.entry_price = 0.0
                    self.trade_history.append((self.current_step, price, "TAKE_PROFIT"))

            elif self.position == -1:  # SHORT
                if price >= self.entry_price * (1 + self.stop_loss_pct):
                    self.balance -= trade_value + fee
                    self.position = 0
                    self.entry_price = 0.0
                    self.trade_history.append((self.current_step, price, "STOP_LOSS"))
                elif price <= self.entry_price * (1 - self.take_profit_pct):
                    self.balance -= trade_value + fee
                    self.position = 0
                    self.entry_price = 0.0
                    self.trade_history.append((self.current_step, price, "TAKE_PROFIT"))

        # Recalcular net worth após stop/take
        self.net_worth = self.balance + (self.position * trade_value)

        # ---- ATUALIZAR MÉTRICAS ----
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth

        # Calcular retorno do passo
        step_return = (self.net_worth - prev_net_worth) / self.initial_balance

        # Janela de retornos para volatilidade
        self.returns_window.append(step_return)

        volatility = np.std(self.returns_window) if len(self.returns_window) > 1 else 0.0

        # ---- REWARD SHARPE-LIKE ----
        alpha = 0.1   # peso do risco
        beta = 0.5    # peso do drawdown

        reward = step_return - alpha * volatility - beta * drawdown

        # ---- REGISTRAR HISTÓRICO ----
        self.equity_history.append(self.net_worth)
        self.drawdown_history.append(drawdown)

        # ---- AVANÇA TEMPO ----
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        next_state = self._get_state()

        return next_state, reward, done

    def _get_state(self):
        state = np.array([
            self.returns[self.current_step],
            self.sma_fast[self.current_step],
            self.sma_slow[self.current_step],
            self.rsi[self.current_step],
            float(self.position),   # -1, 0, 1
            self.balance / self.initial_balance
        ], dtype=np.float32)

        return state
