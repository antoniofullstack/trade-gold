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

        self.reset()

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
        self.returns_window = []

        return self._get_state()

    def step(self, action):
        done = False
        reward = 0.0
        
        # Guardar net worth anterior para calcular retorno
        prev_net_worth = self.net_worth
        
        price = self.data.loc[self.current_step, "Close"]

        # ---- AÇÃO: BUY (1) ----
        if action == 1:
            # abrir long
            if self.position == 0:
                cost = price * self.trade_size
                fee = cost * self.transaction_cost
                if self.balance >= cost + fee:
                    self.balance -= cost + fee
                    self.position = 1
                    self.entry_price = price
                    self.trade_history.append((self.current_step, price, "BUY"))

            # fechar short
            elif self.position == -1:
                profit = (self.entry_price - price) * self.trade_size
                fee = price * self.trade_size * self.transaction_cost
                self.balance += (price * self.trade_size) + profit - fee
                self.position = 0
                self.entry_price = 0.0
                self.trade_history.append((self.current_step, price, "CLOSE_SHORT"))

        # ---- AÇÃO: SELL (2) ----
        elif action == 2:
            # abrir short
            if self.position == 0:
                fee = price * self.trade_size * self.transaction_cost
                self.balance -= fee
                self.position = -1
                self.entry_price = price
                self.trade_history.append((self.current_step, price, "SELL"))

            # fechar long
            elif self.position == 1:
                profit = (price - self.entry_price) * self.trade_size
                fee = price * self.trade_size * self.transaction_cost
                self.balance += (price * self.trade_size) + profit - fee
                self.position = 0
                self.entry_price = 0.0
                self.trade_history.append((self.current_step, price, "CLOSE_LONG"))

        # ---- AÇÃO: CLOSE (3) ----
        elif action == 3 and self.position != 0:
            if self.position == 1:
                profit = (price - self.entry_price) * self.trade_size
            else:
                profit = (self.entry_price - price) * self.trade_size

            fee = price * self.trade_size * self.transaction_cost
            self.balance += (price * self.trade_size) + profit - fee
            self.position = 0
            self.entry_price = 0.0
            self.trade_history.append((self.current_step, price, "CLOSE"))

        # ---- ATUALIZA NET WORTH ----
        if self.position == 1:  # LONG
            unrealized = (price - self.entry_price) * self.trade_size
            self.net_worth = self.balance + price * self.trade_size + unrealized
        elif self.position == -1:  # SHORT
            unrealized = (self.entry_price - price) * self.trade_size
            self.net_worth = self.balance + price * self.trade_size + unrealized
        else:
            self.net_worth = self.balance

        # ---- STOP LOSS E TAKE PROFIT AUTOMÁTICOS ----
        if self.position != 0:
            if self.position == 1:  # LONG
                if price <= self.entry_price * (1 - self.stop_loss_pct):
                    profit = (price - self.entry_price) * self.trade_size
                    fee = price * self.trade_size * self.transaction_cost
                    self.balance += (price * self.trade_size) + profit - fee
                    self.position = 0
                    self.entry_price = 0.0
                    self.trade_history.append((self.current_step, price, "STOP_LOSS"))
                elif price >= self.entry_price * (1 + self.take_profit_pct):
                    profit = (price - self.entry_price) * self.trade_size
                    fee = price * self.trade_size * self.transaction_cost
                    self.balance += (price * self.trade_size) + profit - fee
                    self.position = 0
                    self.entry_price = 0.0
                    self.trade_history.append((self.current_step, price, "TAKE_PROFIT"))

            elif self.position == -1:  # SHORT
                if price >= self.entry_price * (1 + self.stop_loss_pct):
                    profit = (self.entry_price - price) * self.trade_size
                    fee = price * self.trade_size * self.transaction_cost
                    self.balance += (price * self.trade_size) + profit - fee
                    self.position = 0
                    self.entry_price = 0.0
                    self.trade_history.append((self.current_step, price, "STOP_LOSS"))
                elif price <= self.entry_price * (1 - self.take_profit_pct):
                    profit = (self.entry_price - price) * self.trade_size
                    fee = price * self.trade_size * self.transaction_cost
                    self.balance += (price * self.trade_size) + profit - fee
                    self.position = 0
                    self.entry_price = 0.0
                    self.trade_history.append((self.current_step, price, "TAKE_PROFIT"))

        # Recalcular net worth após stop/take
        if self.position == 0:
            self.net_worth = self.balance

        # ---- ATUALIZAR MÉTRICAS ----
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth

        # Calcular retorno do passo
        step_return = (self.net_worth - prev_net_worth) / self.initial_balance

        # Janela de retornos para volatilidade
        self.returns_window.append(step_return)
        if len(self.returns_window) > 20:
            self.returns_window.pop(0)

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
        row = self.data.loc[self.current_step]

        state = np.array([
            row["Return"],
            row["SMA_fast"],
            row["SMA_slow"],
            row["RSI"],
            float(self.position),   # -1, 0, 1
            self.balance / self.initial_balance
        ], dtype=np.float32)

        return state
