# Trade Gold - IA de Trading com Aprendizado por Reforço

Uma aplicação Python completa de trading algorítmico para o mercado de ouro (XAUUSD) utilizando Deep Q-Network (DQN) com aprendizado por reforço.

## 🎯 Visão Geral

Este projeto implementa uma IA que aprende a operar no mercado de ouro através de aprendizado por reforço, incluindo:

- Double DQN com Target Network para estabilidade
- Prioritized Experience Replay para aprendizado eficiente
- Short Selling e Gestão de Risco (Stop Loss/Take Profit)
- Reward Sharpe-like ajustado ao risco
- Validação Walk-Forward para robustez
- Visualizações profissionais de performance

## 📁 Estrutura do Projeto

```
trade-gold/
├── data/                           # Dados históricos
├── env/
│   └── gold_env.py                 # Ambiente de trading (OpenAI Gym style)
├── agent/
│   ├── dqn_agent.py                 # Agente Double DQN com PER
│   └── per_buffer.py                # Prioritized Experience Replay
├── utils/
│   ├── indicators.py                # Indicadores técnicos (RSI, SMA)
│   └── data_processor.py           # Processamento de dados
├── colab/
│   └── colab_setup.py              # Helper de setup no Google Colab
├── train.py                         # Script de treinamento
├── evaluate.py                      # Script de avaliação com visualização
├── walk_forward.py                  # Validação walk-forward
├── requirements.txt                 # Dependências
└── README.md                        # Documentação
```

## 🚀 Instalação (Local)

1. Clone o repositório:
```bash
cd d:\scripts\trade-gold
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

## ☁️ Google Colab (Recomendado)

Use as células abaixo no seu notebook do Colab:

```bash
# 1) Clonar o repositório (substitua pela URL do seu repo)
!git clone <URL_DO_SEU_REPO> trade-gold
%cd /content/trade-gold

# 2) Instalar dependências
!pip -q install -r requirements.txt

# 3) (Opcional) Setup helper
from colab.colab_setup import setup_repo
setup_repo("/content/trade-gold", install_deps=False)
```

Para usar seus próprios dados:

```bash
# Upload do CSV via interface do Colab (ou Google Drive)
# Depois aponte o caminho, por exemplo:
python train.py --data data/xauusd.csv --episodes 200
```

## 📊 Formato dos Dados

O sistema espera dados em formato CSV com as seguintes colunas:
- `Close`: Preço de fechamento do ouro

Colunas adicionais serão calculadas automaticamente:
- `Return`: Retornos percentuais
- `SMA_fast`: Média móvel curta (10 períodos)
- `SMA_slow`: Média móvel longa (30 períodos)
- `RSI`: Relative Strength Index (14 períodos)

## 🎮 Como Usar

### 1. Treinamento

Treine a IA com dados históricos:

```bash
# Usar dados existentes
python train.py --data data/xauusd.csv --episodes 1000

# Criar dados amostra para teste
python train.py --episodes 500 --model-dir models

# Desabilitar Prioritized Experience Replay
python train.py --no-per --episodes 1000
```

Parâmetros principais:
- `--data`: Caminho para arquivo CSV de dados
- `--episodes`: Número de episódios de treinamento
- `--model-dir`: Diretório para salvar modelos
- `--lr`: Learning rate (default: 1e-3)
- `--batch-size`: Tamanho do batch (default: 64)

### 2. Avaliação

Avalie o modelo treinado:

```bash
# Avaliar modelo
python evaluate.py --model models/model_final.pth --data data/xauusd.csv

# Gerar visualizações
python evaluate.py --model models/model_final.pth --episodes 5 --plots --plot-dir plots
```

### 3. Validação Walk-Forward

Valide a robustez do modelo:

```bash
# Validação walk-forward completa
python walk_forward.py --data data/xauusd.csv --train-window 2000 --test-window 500

# Salvar modelos de cada janela
python walk_forward.py --data data/xauusd.csv --save-models --model-dir wf_models
```

## 🧠 Componentes Principais

### Ambiente de Trading (`gold_env.py`)

- Ações: Hold(0), Buy(1), Sell(2), Close(3)
- Posições: Flat(0), Long(1), Short(-1)
- Reward: step_return - α*volatility - β*drawdown
- Risco: Stop loss (1%), Take profit (2%), custo (0.1%)

### Agente Double DQN (`dqn_agent.py`)

- Rede: MLP com 2 camadas ocultas (64 neurônios)
- Target Network: Atualização a cada 500 passos
- Exploração: ε-greedy com decaimento
- PER: α=0.6, β=0.4→1.0

### Validação Walk-Forward

- Janelas: Treino 2000 candles, Teste 500 candles
- Passo: 500 candles entre blocos
- Métricas: Win rate, Sharpe ratio, Max drawdown

## 📈 Métricas de Performance

O sistema avalia a performance através de:

- Retorno Total: Lucro/prejuízo percentual
- Sharpe Ratio: Retorno ajustado ao risco
- Maximum Drawdown: Maior queda do capital
- Win Rate: Percentual de janelas lucrativas
- Número de Trades: Frequência de operações

## ⚙️ Parâmetros Configuráveis

### Ambiente
- `initial_balance`: Capital inicial (default: 10,000)
- `trade_size`: Tamanho da operação (default: 1.0)
- `transaction_cost`: Custo de transação (default: 0.001)
- `stop_loss_pct`: Stop loss percentual (default: 0.01)
- `take_profit_pct`: Take profit percentual (default: 0.02)

### Agente
- `lr`: Learning rate (default: 1e-3)
- `gamma`: Fator de desconto (default: 0.99)
- `epsilon_decay`: Decaimento da exploração (default: 0.995)
- `batch_size`: Tamanho do batch (default: 64)

## 🎯 Resultados Esperados

Uma IA bem treinada deve apresentar:

- Sharpe Ratio > 1.0 em walk-forward
- Max Drawdown < 20% consistentemente
- Win Rate > 70% nas janelas de teste
- Equity Curve suave e crescente
- Número razoável de trades (sem overtrading)

## ⚠️ Avisos Importantes

1. Risco Real: Este é um projeto educacional. Não use dinheiro real sem testes extensivos.
2. Overfitting: Sempre valide com walk-forward antes de confiar no modelo.
3. Dados de Qualidade: Garanta que os dados históricos sejam confiáveis.
4. Mudanças de Mercado: O modelo pode precisar retreinamento periódico.

## 🔧 Personalização

### Adicionar Novos Indicadores

Edite `utils/indicators.py`:

```python
def calculate_new_indicator(prices, window=20):
    # Seu cálculo aqui
    return indicator_values

# Em add_technical_indicators():
df['NewIndicator'] = calculate_new_indicator(df[price_col])
```

### Modificar Reward Function

Edite `env/gold_env.py` na seção de reward:

```python
# Exemplo: reward com penalidade por trades excessivos
trade_penalty = 0.01 if len(self.trade_history) > prev_trades else 0
reward = step_return - alpha*volatility - beta*drawdown - trade_penalty
```

## 📚 Referências

- Deep Q-Network (Mnih et al., 2015)
- Double Q-Learning (van Hasselt et al., 2016)
- Prioritized Experience Replay (Schaul et al., 2015)
- Walk-Forward Analysis (Pardo, 2008)

## 🤝 Contribuições

Contribuições são bem-vindas! Principais áreas de melhoria:

- Novos algoritmos (PPO, SAC)
- Mais indicadores técnicos
- Otimização de hiperparâmetros
- Interface web para monitoramento

## 📄 Licença

Este projeto é para fins educacionais. Use por sua conta e risco.

---

Desenvolvido com base em research quant profissional e melhores práticas de trading algorítmico.
