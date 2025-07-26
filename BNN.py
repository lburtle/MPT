import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoNormal
from pyro.optim import ClippedAdam
import numpy as np
import yfinance as yf
import pandas as pd
import datetime
from datetime import datetime, timedelta
import time
import json

if torch.cuda.is_available():
    device = 'cuda'
    print('CUDA available. Using GPU')
else:
    device = 'cpu'
    print('CUDA unavailable. Using CPU')

torch.manual_seed(0)
np.random.seed(0)
pyro.set_rng_seed(0)

ticker_symbol = 'TSLA' ## Change this for each stock to analyze

def get_stock_data(ticker=ticker_symbol, start_date=None, end_date=None):
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=7*365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    print(f"Downloading {ticker} data from {start_date} to {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker} in the specified date range.")
    
    if isinstance(df.columns, pd.MultiIndex):
        print(f"Original MultiIndex columns: {df.columns}")
        if all(col[1] == ticker for col in df.columns):
            df.columns = df.columns.droplevel(1)
            print(f"Columns after droplevel(1): {df.columns}")
        else:
            df.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]
            print(f"Flattened columns: {df.columns}")
    


    new_columns = []
    for col in df.columns:
        col = col.replace(' ', '_').replace('.', '_').replace('-', '_')
        if col.lower() == 'adj_close':
            new_columns.append('adj_close')
        elif col.lower() == f'adj_close_{ticker.lower()}':
            new_columns.append('Close')
        elif col.lower() == f'close_{ticker.lower()}':
            new_columns.append('Close')
        elif col.lower() == f"volume_{ticker.lower()}":
            new_columns.append('Volume')
        else:
            new_columns.append(col)

    df.columns = new_columns
    print(f"Columns after initial processing and renaming: {df.columns.tolist()}")

    if 'Close' not in df.columns:
        possible_close_cols = ['Adj_Close', 'Close']
        found_close = False
        for c in possible_close_cols:
            if c in df.columns:
                df['Close'] = pd.to_numeric(df[c], errors='coerce')
                df.drop(columns=[c], inplace=True, errors='ignore')
                found_close = True
                print(f"Found and used '{c}' as 'Close' column.")
                break
            elif f'{c}_{ticker}' in df.columns:
                df['Close'] = pd.to_numeric(df[f'{c}_{ticker}'], errors='coerce')
                df.drop(columns=[f'{c}_{ticker}'], inplace=True, errors='ignore')
                found_close = True
                print(f"Found and used '{c}_{ticker}' as 'Close' column.")
                break
        if not found_close:
            raise ValueError(f"'Close' or 'Adj_Close' column (or prefixed version) not found after extensive processing. Available columns: {df.columns.tolist()}")
    else:
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    if 'Volume' not in df.columns:
        possible_volume_cols = ['Volume']
        found_volume = False
        for c in possible_volume_cols:
            if c in df.columns:
                df['Volume'] = pd.to_numeric(df[c], errors='coerce')
                df.drop(columns=[c], inplace=True, errors='ignore')
                found_volume = True
                print(f"Found and used '{c}' as 'Volume' columns.")
                break
            elif f'{c}_{ticker}' in df.columns:
                df['Volume'] = pd.to_numeric(df[f'{c}_{ticker}'], errors='coerce')
                df.drop(columns=[f'{c}_{ticker}'], inplace=True, errors='ignore')
                found_volume = True
                print(f"Found and used '{c}_{ticker}' as 'Volume' columns.")
                break
        if not found_volume:
            print(f"Warning: 'Volume column not found in download data after extensive processing. Filling with zeros.")
            df['Volume'] = 0.0
    else:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    return df

def prepare_stock_data(df, lookback_window=5, prediction_horizon=1):
    df = df.copy()

    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column for stock prices.")
    if isinstance(df['Close'], pd.DataFrame):
        df['Close'] = df['Close'].squeeze()

    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + RS))
    df['RSI_14'].fillna(50, inplace=True)

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)

    df['Price_Change'] = df['Close'].diff() / df['Close'].shift(1)
    
    df['Day_of_Week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['Day_of_Week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

    df['Avg_Volume_5'] = df['Volume'].rolling(window=5).mean()

    df['Relative_Volume'] = (df['Volume'].squeeze() / df['Avg_Volume_5'].squeeze()).replace([np.inf, -np.inf], np.nan)

    feature_cols = [
        'Close',
        'SMA_5',
        'SMA_10',
        'RSI_14',
        'MACD_Hist',
        'BB_Upper',
        'BB_Lower',
        'Price_Change',
        'Relative_Volume',
        'Day_of_Week_sin',
        'Day_of_Week_cos'
    ]

    initial_rows = len(df)
    df.dropna(inplace=True)
    rows_dropped = initial_rows - len(df)
    print(f"Dropped {rows_dropped} rows due to NaN values in fautre calculations.")

    for col in feature_cols:
        col_series = df[col]
        if isinstance(col_series, pd.DataFrame):
            print(f"Warning: Column '{col}' is a DataFrame. Squeezing it to Series.")
            col_series = col_series.squeeze()
            if isinstance(col_series, pd.DataFrame):
                raise TypeError(f"Column '{col}' is still a DataFrame after squeezing. Please check the data.")
            
        if col_series.isnull().any():
            print(f"Warning: Column '{col}' contains NaN values after processing. Filling with mean.")
            col_series.fillna(col_series.mean(), inplace=True)
            df[col] = col_series

    X_list = []
    Y_list = []
    dates_list = []
    last_prices_list = []

    if len(df) < lookback_window + prediction_horizon:
        raise ValueError(f"Not enough data after feature engineering. Need at least {lookback_window + prediction_horizon} rows, but got {len(df)}.")
    
    for i in range(len(df) - lookback_window - prediction_horizon + 1):
        current_features = df[feature_cols].iloc[i : i + lookback_window].values.flatten()
        X_list.append(current_features)

        last_close_price = df['Close'].iloc[i + lookback_window - 1]
        target_price = df['Close'].iloc[i + lookback_window + prediction_horizon - 1]
        if isinstance(target_price, (pd.Series, pd.DataFrame)):
            target_price = target_price.squeeze()

        if last_close_price != 0:
            target_return = (target_price - last_close_price) / last_close_price
        else:
            target_return = 0.0
        Y_list.append(target_return)
        last_prices_list.append(last_close_price)

        dates_list.append(df.index[i + lookback_window + prediction_horizon - 1])

    X = np.array(X_list)
    Y = np.array(Y_list)
    last_prices = np.array(last_prices_list)

    print(f"Calculated input_dim for BNN: {X.shape[1]}")

    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
        dates_list,
        {'y_min': Y.min(), 'y_max': Y.max(), 'x_min': X.min(axis=0), 'x_max': X.max(axis=0)},
        X.shape[1],
        torch.tensor(last_prices, dtype=torch.float32)
    )

class HybridBNN(PyroModule):
    def __init__(self, hidden_size=100, input_dim=110):
        super().__init__()
        print(f"Initializing HybridBNN with input_dim={input_dim} and hidden_size={hidden_size}")
        
        self.fc1 = torch.nn.Linear(input_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.fc3 = PyroModule[nn.Linear](hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        self.fc_out = PyroModule[nn.Linear](hidden_size, 1)

        self.fc3.weight = PyroSample(dist.Normal(
            torch.tensor(0., device=device),
            torch.tensor(1.0, device=device)
        ).expand([hidden_size, hidden_size]).to_event(2))
        self.fc3.bias = PyroSample(dist.Normal(
            torch.tensor(0., device=device),
            torch.tensor(1.0, device=device)
        ).expand([hidden_size]).to_event(1))

        self.fc_out.weight = PyroSample(dist.Normal(
            torch.tensor(0., device=device),
            torch.tensor(1.0, device=device)
        ).expand([1, hidden_size]).to_event(2))
        self.fc_out.bias = PyroSample(dist.Normal(
            torch.tensor(0., device=device),
            torch.tensor(1.0, device=device)
        ).expand([1]).to_event(1))

        self.activation = nn.ReLU()

    def forward(self, x, y=None):
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.activation(self.ln3(self.fc3(x)))

        mu = self.fc_out(x).squeeze(-1)

        sigma = pyro.sample("sigma", dist.Uniform(
            torch.tensor(0.05, device=device),
            torch.tensor(0.5, device=device)
        ))

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu
    
def calculate_rmse(y_true_normalized, y_pred_normalized, y_min, y_max):
    y_true_unnormalized = y_true_normalized * (y_max - y_min) + y_min
    y_pred_unnormalized = y_pred_normalized * (y_max - y_min) + y_min
    return torch.sqrt(torch.mean((y_true_unnormalized - y_pred_unnormalized) ** 2)).item()

def calculate_single_row_features(df_past, feature_cols, lookback_window=10):
    df_single = df_past.copy()

    required_for_features = ['Close', 'Volume']
    for col in required_for_features:
        if col not in df_single.columns:
            df_single[col] = 0.0
            print(f"Warning: Column '{col}' not found in df_past. Added with 0.0")
        
        if not df_single[col].empty:
            df_single[col] = pd.to_numeric(df_single[col], errors='coerce')
            df_single[col].fillna(method='ffill', inplace=True)
            df_single[col].fillna(method='bfill', inplace=True)
            df_single[col].fillna(df_single[col].mean() if not df_single[col].empty else 0.0, inplace=True)
            df_single[col].fillna(0.0, inplace=True)
        else:
            df_single[col] = pd.Series([0.0] * len(df_single.index), index=df_single.index, dtype=float)
            print(f"Warning: Column '{col}' is empty in df_past. Filled with zeros.")

    df_single['SMA_5'] = df_single['Close'].rolling(window=5, min_periods=1).mean().fillna(df_single['Close'])
    df_single['SMA_10'] = df_single['Close'].rolling(window=10, min_periods=1).mean().fillna(df_single['Close'])

    delta = df_single['Close'].diff().fillna(0)

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    gain_mean = gain.rolling(window=14, min_periods=1).mean().fillna(0)
    loss_mean = loss.rolling(window=14, min_periods=1).mean().fillna(0)

    RS = gain_mean / loss_mean.replace(0, np.nan)
    df_single['RSI_14'] = 100 - (100 / (1 + RS))
    df_single['RSI_14'].fillna(50, inplace=True)

    exp1 = df_single['Close'].ewm(span=12, adjust=False, min_periods=1).mean().fillna(df_single['Close'])
    exp2 = df_single['Close'].ewm(span=26, adjust=False, min_periods=1).mean().fillna(df_single['Close'])
    df_single['MACD'] = (exp1 - exp2).fillna(0)
    df_single['Signal_Line'] = df_single['MACD'].ewm(span=9, adjust=False, min_periods=1).mean().fillna(0)
    df_single['MACD_Hist'] = (df_single['MACD'] - df_single['Signal_Line']).fillna(0)

    df_single['BB_Mid'] = df_single['Close'].rolling(window=20, min_periods=1).mean().fillna(df_single['Close'])
    df_single['BB_Std'] = df_single['Close'].rolling(window=20, min_periods=1).std().fillna(0)
    df_single['BB_Upper'] = (df_single['BB_Mid'] + (df_single['BB_Std'] * 2)).fillna(df_single['BB_Mid'])
    df_single['BB_Lower'] = (df_single['BB_Mid'] - (df_single['BB_Std'] * 2)).fillna(df_single['BB_Mid'])

    df_single['Price_Change'] = (df_single['Close'].diff() / df_single['Close'].shift(1)).fillna(0)

    if not isinstance(df_single.index, pd.DatetimeIndex):
        df_single.index = pd.to_datetime(df_single.index)
    df_single['Day_of_Week_sin'] = np.sin(2 * np.pi * df_single.index.dayofweek / 7)
    df_single['Day_of_Week_cos'] = np.cos(2 * np.pi * df_single.index.dayofweek / 7)

    df_single['Avg_Volume_5'] = df_single['Volume'].rolling(window=5, min_periods=1).mean().fillna(df_single['Volume'])
    df_single['Relative_Volume'] = (df_single['Volume'] / df_single['Avg_Volume_5']).replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in feature_cols:
        if col not in df_single.columns:
            df_single[col] = 0.0
            print(f"Warning: Missing required feature column '{col}'. Added with 0.0")

        if not df_single[col].empty:
            df_single[col] = pd.to_numeric(df_single[col], errors='coerce')
            df_single[col].fillna(df_single[col].mean() if not df_single[col].empty else 0.0, inplace=True)
            df_single[col].fillna(0.0, inplace=True)
        else:
            df_single[col] = pd.Series([0.0] * len(df_single.index), index=df_single.index, dtype=float)
            print(f"Warning: Column '{col}' is empty in df_past. Filled with zeros.")

    if len(df_single) < lookback_window:
        print(f"Warning: df_single has {len(df_single)} rows, less than lookback_window={lookback_window}. Padding with zeros.")
        if not df_single[feature_cols].empty:
            temp_df_for_padding = df_single[feature_cols].copy()
            for col in feature_cols:
                if temp_df_for_padding[col].isnull().any():
                    temp_df_for_padding[col].fillna(temp_df_for_padding[col].mean() if not temp_df_for_padding[col].empty else 0.0, inplace=True)
                    temp_df_for_padding[col].fillna(0.0, inplace=True)

            last_valid_row = temp_df_for_padding.iloc[-1].values
            current_features = np.tile(last_valid_row, (lookback_window, 1)).flatten()
        else:
            print("Error: df_single is empty and cannot be padded. Returning to zeros.")
            current_features = np.zeros(len(feature_cols) * lookback_window)
    else:
        current_features = df_single[feature_cols].iloc[-lookback_window:].values.flatten()
    return current_features

def init_loc_fn(site):
    param_shape = site['fn'].sample().shape

    if site['name'].endswith('weight'):
        val = torch.empty(param_shape, device=device)
        nn.init.kaiming_uniform_(val, nonlinearity='leaky_relu')
        return val
    if site['name'].endswith('bias'):
        val = torch.zeros(param_shape, device=device)
        return val
    
    if site['name'] == 'sigma':
        print(f"init_loc_fn: Initializing sigma to 0.2")
        return torch.tensor(0.2, device=device)
    
    if 'scale_tril' in site['name'] or 'scale' in site['name'] or 'unconstrained_scale' in site['name']:
        return torch.full(param_shape, 0.01, device=device)
    
    return torch.zeros(param_shape, device=device)

def train_model(ticker_symbol):
    stock_df = get_stock_data(ticker_symbol, start_date='2018-01-01', end_date=datetime.now().strftime('%Y-%m-%d'))

    lookback_window = 10 ## Could change this for different lookback weighting (e.g. decay for larger window size)
    prediction_horizon = 3 ## Number of days to predict into the future

    x_data_full_unnorm, y_data_full_unnorm, dates_full, _, input_dim_determined, last_prices_full = \
    prepare_stock_data(stock_df, lookback_window=lookback_window, prediction_horizon=prediction_horizon)

    print(f"Main: input_dim from prepare_stock_data: {input_dim_determined}")

    train_split = int(0.8 * len(x_data_full_unnorm))

    x_data_train_unnorm = x_data_full_unnorm[:train_split]
    y_data_train_unnorm = y_data_full_unnorm[:train_split]
    x_data_test_unnorm = x_data_full_unnorm[train_split:]
    y_data_test_unnorm = y_data_full_unnorm[train_split:]
    dates_test = dates_full[train_split:]
    last_prices_test = last_prices_full[train_split:]

    x_min_train = x_data_train_unnorm.min(axis=0).values
    x_max_train = x_data_train_unnorm.max(axis=0).values
    y_min_train = y_data_train_unnorm.min().item()
    y_max_train = y_data_train_unnorm.max().item()

    denominator_x_train = (x_max_train - x_min_train)
    denominator_x_train[denominator_x_train == 0] = 1.0

    denominator_y_train = (y_max_train - y_min_train)

    if denominator_y_train == 0:
        print("Warning: Training target returns (Y) are constant. Normalizing to 0.5")
        y_data_train = torch.full_like(y_data_train_unnorm, 0.5)
        y_data_test = torch.full_like(y_data_test_unnorm, 0.5)
    else:
        y_data_train = (y_data_train_unnorm - y_min_train) / denominator_y_train
        y_data_test = (y_data_test_unnorm - y_min_train) / denominator_y_train

    x_data_train = (x_data_train_unnorm - x_min_train) / denominator_x_train
    x_data_test = (x_data_test_unnorm - x_min_train) / denominator_x_train

    scaling_info = {'x_min': x_min_train, 'x_max': x_max_train,
                    'y_min': y_min_train, 'y_max': y_max_train}

    x_data_train = x_data_train.to(device)
    y_data_train = y_data_train.to(device)
    x_data_test = x_data_test.to(device)
    y_data_test = y_data_test.to(device)

    scaling_info['x_min'] = scaling_info['x_min'].to(device)
    scaling_info['x_max'] = scaling_info['x_max'].to(device)
    scaling_info['y_min'] = torch.tensor(scaling_info['y_min'], device=device)
    scaling_info['y_max'] = torch.tensor(scaling_info['y_max'], device=device)

    print(f"Training data size: {len(x_data_train)}")
    print(f"Test data size: {len(x_data_test)}")

    bnn = HybridBNN(hidden_size=100, input_dim=input_dim_determined).to(device)

    guide = AutoNormal(bnn, init_loc_fn=init_loc_fn)
    guide = guide.to(device)

    pyro.clear_param_store()

    optimizer = ClippedAdam({'lr': 0.005, 'clip_norm': 10.0, 'lrd': 0.9995})

    elbo_loss = Trace_ELBO(num_particles=1)
    svi = SVI(bnn, guide, optimizer, loss=elbo_loss)

    print("Initializing guide...")
    guide(x_data_train, y_data_train)
    print("Guide initialized.")

    print(f"Model device: {next(bnn.parameters()).device}")
    print(f"Guide device: {next(guide.parameters()).device}")
    print(f"Data device: {x_data_train.device}, {y_data_train.device}")

    num_steps = 8500
    annealing_start_step = 500
    annealing_end_step = 6500
    final_kl_weight = 1.0

    losses = []

    for step in range(num_steps):
        step_start_time = time.time()

        if step < annealing_start_step:
            kl_weight = 0.0
        elif step >= annealing_end_step:
            kl_weight = final_kl_weight
        else:
            kl_weight = final_kl_weight * ((step - annealing_start_step) / (annealing_end_step - annealing_start_step))

        elbo_loss.beta = kl_weight

        try:
            loss = svi.step(x_data_train, y_data_train)
            step_time = time.time() - step_start_time

            if step % 100 == 0:
                print(f"Step {step} completed in {step_time:.4f}s, Loss: {loss:.4f}, KL Weight: {kl_weight:.4f}")
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break
        
        losses.append(loss)

        if step % 100 == 0 or step == num_steps - 1:
            plot_start_time = time.time()

            bnn.eval()
            with torch.no_grad():
                try:
                    if "sigma_unconstrained" in guide.prototype_trace.nodes:
                        try:
                            sigma_loc = pyro.param(f"AutoNormal.locs.sigma_unconstrained").item()
                            sigma_scale_tril_diag = pyro.param(f"AutoNormal.scales.sigma_unconstrained_scale_tril").diag().item()
                            try:
                                sigma_loc_guide = pyro.param("AutoNormal.locs.sigma_unconstrained").item()
                                sigma_scale_guide = pyro.param("AutoNormal.scales.sigma_unconstrained_scale_tril").diag().item()
                                print(f"Guide's sigma_unconstrained parameters: loc={sigma_loc_guide:.6f}, scale_tril_diag={sigma_scale_guide:.6f}")
                            except Exception as e_guide_param:
                                print(f"Could not retrieve guide's sigma paramteres: {e_guide_param}")

                            predictive = Predictive(bnn, guide=guide, num_samples=500, return_sites=("obs", "_RETURN", "sigma"))
                            samples = predictive(x_data_test)

                            sigma_samples = samples['sigma']
                            print(f"Predictive Sigma samples (mean/std): {sigma_samples.mean().item():.6f}/{sigma_samples.std().item():.6f}")

                            predicted_returns_normalized = samples["_RETURN"].mean(0)
                            std_pred_normalized = samples["_RETURN"].std(0)

                            y_min_tensor = scaling_info['y_min']
                            y_max_tensor = scaling_info['y_max']

                            if denominator_y_train == 0:
                                predicted_returns_unnormalized = y_min_tensor
                                actual_retuns_unnormalized = y_min_tensor
                                std_pred_unnormalized_returns = torch.zeros_like(std_pred_normalized)
                            else:
                                predicted_returns_unnormalized = (predicted_returns_normalized * (y_max_tensor - y_min_tensor)) + y_min_tensor
                                actual_returns_unnormalized = y_data_test * (y_max_tensor - y_min_tensor) + y_min_tensor
                                std_pred_unnormalized_returns = std_pred_normalized * (y_max_tensor - y_min_tensor)
                            
                            mean_pred_unnormalized_prices = last_prices_test.to(device) * (1 + predicted_returns_unnormalized)
                            actual_unnormalized_prices = last_prices_test.to(device) * (1 + actual_returns_unnormalized)
                            
                            std_pred_unnormalized_prices = std_pred_unnormalized_returns * last_prices_test.to(device)

                            rmse = torch.sqrt(torch.mean((actual_unnormalized_prices - mean_pred_unnormalized_prices) ** 2)).item()


                        except Exception as e_predictive:
                            print(f"Error during predictive sampling at step {step}: {e_predictive}")
                            pass
                    else:
                        print(f"Skipping sigma monitoring as 'sigma_unconstrained' not found in guide's trace nodes.")
                        
                        predictive = Predictive(bnn, guide=guide, num_samples=500, return_sites=("_RETURN",))
                        samples = predictive(x_data_test)

                        predicted_returns_normalized = samples["_RETURN"].mean(0)
                        std_pred_normalized = samples["_RETURN"].std(0)

                        y_min_tensor = scaling_info['y_min']
                        y_max_tensor = scaling_info['y_max']

                        if denominator_y_train == 0:
                                predicted_returns_unnormalized = y_min_tensor
                                actual_retuns_unnormalized = y_min_tensor
                                std_pred_unnormalized_returns = torch.zeros_like(std_pred_normalized)
                        else:
                            predicted_returns_unnormalized = (predicted_returns_normalized * (y_max_tensor - y_min_tensor)) + y_min_tensor
                            actual_returns_unnormalized = y_data_test * (y_max_tensor - y_min_tensor) + y_min_tensor
                            std_pred_unnormalized_returns = std_pred_normalized * (y_max_tensor - y_min_tensor)
                        
                        mean_pred_unnormalized_prices = last_prices_test.to(device) * (1 + predicted_returns_unnormalized)
                        actual_unnormalized_prices = last_prices_test.to(device) * (1 + actual_returns_unnormalized)
                        
                        std_pred_unnormalized_prices = std_pred_unnormalized_returns * last_prices_test.to(device)

                        rmse = torch.sqrt(torch.mean((actual_unnormalized_prices - mean_pred_unnormalized_prices) ** 2)).item()

                except Exception as e_plot:
                    print(f"Error during plotting at step {step}: {e_plot}")
                finally:
                    bnn.train()
    return bnn, guide, scaling_info, lookback_window
                

if __name__ == "__main__":
    bnn, guide, scaling_info, lookback_window = train_model(ticker_symbol=ticker_symbol)

print("Training completed.")

today_date = (datetime.now()).strftime('%Y-%m-%d')
recent_data_start = (datetime.now() - timedelta(days=lookback_window + 6)).strftime('%Y-%m-%d')
latest_stock_df = get_stock_data(ticker_symbol, start_date=recent_data_start, end_date=today_date)

last_actual_close_price = latest_stock_df['Close'].iloc[-1]
print(f"Last actual close price for {ticker_symbol} on {latest_stock_df.index[-1]}: {last_actual_close_price:.2f}")

if len(latest_stock_df) < lookback_window:
    print(f"Not enough data to form a {lookback_window}-day window.")
    print("Cannot make predictions without sufficient historical data.")
else:
    features_cols_for_future = [
        'Close', 'SMA_5', 'SMA_10', 'RSI_14', 'MACD_Hist',
        'BB_Upper', 'BB_Lower', 'Price_Change', 'Relative_Volume',
        'Day_of_Week_sin', 'Day_of_Week_cos'
    ]

input_for_future_features_np = calculate_single_row_features(
        latest_stock_df, features_cols_for_future
)
print(f"Shape of input_for_future_features_np: {input_for_future_features_np.shape}")
print(f"Shape of scaling_info['x_min'].cpu().numpy(): {scaling_info['x_min'].cpu().numpy().shape}")
print(f"Shape of scaling_info['x_max'].cpu().numpy(): {scaling_info['x_max'].cpu().numpy().shape}")

input_for_future_norm = (input_for_future_features_np - scaling_info['x_min'].cpu().numpy()) / scaling_info['x_max'].cpu().numpy()
input_for_future_tensor = torch.tensor(input_for_future_norm, dtype=torch.float32).to(device).unsqueeze(0)

bnn.eval()
with torch.no_grad():
    predictive_future = Predictive(bnn, guide=guide, num_samples=500, return_sites=("_RETURN",))
    samples_future = predictive_future(input_for_future_tensor)

    mean_pred_normalized_future = samples_future["_RETURN"].mean(0)
    std_pred_normalized_future = samples_future["_RETURN"].std(0)

y_min, y_max = scaling_info['y_min'], scaling_info['y_max']
predicted_return_unnorm = mean_pred_normalized_future * (y_max - y_min) + y_min
pred_std_return_unnorm = std_pred_normalized_future * (y_max - y_min)

predicted_close_for_future = last_actual_close_price * (1 + predicted_return_unnorm.item())

lower_bound_price = last_actual_close_price * (1 + predicted_return_unnorm.item() - 2 * pred_std_return_unnorm.item())
upper_bound_price = last_actual_close_price * (1 + predicted_return_unnorm.item() + 2 * pred_std_return_unnorm.item())

print(f"Predicted return for future (unnormalized): {predicted_return_unnorm.item():.4f}")

next_trading_day = latest_stock_df.index[-1] + timedelta(days=3) ## days = # of days ahead to predict
while next_trading_day.weekday() >= 5:
    next_trading_day += timedelta(days=1)

print(f"Predicted close price for {ticker_symbol} on {next_trading_day.strftime('%Y-%m-%d')}: {predicted_close_for_future:.2f}")
print(f"Lower bound price: {lower_bound_price:.2f}, Upper bound price: {upper_bound_price:.2f}")
print(f"Uncertainty (approx. += 2 std dev of Return): +={2 * pred_std_return_unnorm.item():.4f} on return scale, or +={2 * pred_std_return_unnorm.item() * last_actual_close_price:.2f} on price scale")

def get_signal():

    ## Very simplistic strategy: if rising -> buy, if stable -> hold, if dropping -> sell

    if predicted_close_for_future > last_actual_close_price:
        signal = 2
    elif predicted_close_for_future == last_actual_close_price:
        signal = 1
    else:
        signal = 0
    return signal

result = {
    "ticker": ticker_symbol,
    "next_trading_day": next_trading_day.strftime('%Y-%m-%d'),
    "predicted_close": predicted_close_for_future,
    "lower_bound": lower_bound_price,
    "upper_bound": upper_bound_price,
    "signal": get_signal()
}

with open("/output/signal.json", "w") as f:
    json.dump(result, f, indent=2)