import os
os.environ['OMP_NUM_THREADS'] = '1'
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import talib
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

class MarketUnderstandingBot:
    def __init__(self):
        self.config = {
            'account': 333001799,
            'password': "Iamsuccessful1/",
            'server': "XMGlobal-MT5 9",
            'target_profit_usd': 1000,
            'risk_percent': 1,
            'symbols': ["GOLD", "BTCUSD", "US100Cash"],
            'timeframes': [mt5.TIMEFRAME_M30, mt5.TIMEFRAME_H1],
            'max_trades_per_symbol': 1,
            'learning_period': 1460,  # Days of data to learn from
            'confidence_threshold': 0.65,
            'risk_reward_ratio': 2.0,
            'volatility_lookback': 14,
            'adaptive_trailing_factor': 0.4,
            'min_trail_distance': 1.5,
            'max_trail_distance': 4.0,
            'market_regime_lookback': 63,
            'model_retrain_interval': 7,  # Days
            'prediction_horizon': 5,  # Number of candles ahead to predict
            'min_trade_confidence': 0.03,  # Minimum model confidence to trade
            'min_historical_win_rate': 0.55,  # Minimum historical win rate to trade
            'min_historical_samples': 5  # Minimum historical similar scenarios to consider
        }
        
        self.symbol_info = {}
        self.market_models = {}  # Stores trained ML models
        self.market_knowledge = {}  # Stores learned market behaviors
        self.active_trades = {}
        self.total_profit = 0
        self.target_hit = False
        self.last_target_hit_time = None
        self.initialize_mt5()
        self._initialize_symbols()
        self.learn_market_dynamics()

    def initialize_mt5(self):
        if not mt5.initialize():
            print("MT5 initialization failed")
            mt5.shutdown()
            raise Exception("MT5 initialization failed")
        
        authorized = mt5.login(
            self.config['account'],
            password=self.config['password'],
            server=self.config['server']
        )
        
        if not authorized:
            print("Login failed")
            mt5.shutdown()
            raise Exception("Login failed")
        
        print(f"Connected to account #{self.config['account']} at {self.config['server']}")

    def _initialize_symbols(self):
        print("\nInitializing symbols...")
        for symbol in self.config['symbols']:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"  [!] Symbol {symbol} not found, skipping")
                continue
            
            point = symbol_info.point
            digits = symbol_info.digits
            
            pip_value = 0.0001  # Default for most pairs
            if digits == 3 or digits == 2:  # JPY pairs and similar
                pip_value = 0.01
            elif digits == 5:  # Some brokers use 5 digits
                pip_value = 0.0001
            
            contract_size = symbol_info.trade_contract_size
            tick_value = symbol_info.trade_tick_value
            min_stop_distance = symbol_info.trade_stops_level * point
            min_stop_pips = min_stop_distance / pip_value if pip_value != 0 else 0
            current_spread = symbol_info.spread * point
            spread_pips = current_spread / pip_value if pip_value != 0 else 0
            
            self.symbol_info[symbol] = {
                'point': point,
                'digits': digits,
                'pip_value': pip_value,
                'tick_value': tick_value,
                'contract_size': contract_size,
                'trade_allowed': symbol_info.trade_mode in [mt5.SYMBOL_TRADE_MODE_FULL, 
                                                        mt5.SYMBOL_TRADE_MODE_LONGONLY, 
                                                        mt5.SYMBOL_TRADE_MODE_SHORTONLY],
                'tick_size': symbol_info.trade_tick_size,
                'spread': symbol_info.spread,
                'current_spread_pips': spread_pips,
                'min_lot': symbol_info.volume_min,
                'max_lot': symbol_info.volume_max,
                'lot_step': symbol_info.volume_step,
                'min_stop_distance': min_stop_distance,
                'min_stop_pips': min_stop_pips,
                'freeze_level': symbol_info.trade_freeze_level * point,
                'trade_mode': symbol_info.trade_mode,
            }
            
            market_open = "OPEN" if self._is_market_open(symbol) else "CLOSED"
            trade_status = "TRADABLE" if self.symbol_info[symbol]['trade_allowed'] else "NOT TRADABLE"
            print(f"  [i] {symbol}: Market {market_open}, Status: {trade_status}")
            print(f"      Digits: {digits}, Point: {point}, Pip Value: {pip_value}")
            print(f"      Tick Value: {tick_value}, Contract Size: {contract_size}")
            print(f"      Min Stop Distance: {min_stop_distance} ({min_stop_pips:.1f} pips)")
            print(f"      Current Spread: {spread_pips:.1f} pips")

    def _is_market_open(self, symbol):
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return False
        return symbol_info.trade_mode in [mt5.SYMBOL_TRADE_MODE_FULL, 
                                         mt5.SYMBOL_TRADE_MODE_LONGONLY, 
                                         mt5.SYMBOL_TRADE_MODE_SHORTONLY]

    def get_historical_data(self, symbol, timeframe, start_date, end_date=None, max_retries=3):
        if end_date is None:
            end_date = datetime.now()
        
        # Retry mechanism
        for attempt in range(max_retries):
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
            if rates is not None and len(rates) > 0:
                break
                
            print(f"Attempt {attempt + 1} failed for {symbol} on {timeframe}")
            time.sleep(1)  # Wait before retrying
        
        if rates is None or len(rates) == 0:
            print(f"Warning: Failed to get historical data for {symbol} on {timeframe} after {max_retries} attempts")
            return None
        
        # Rest of your processing...
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        if 'volume' not in df.columns:
            df['volume'] = 0
            
        # Calculate price features
        df['returns'] = df['close'].pct_change().fillna(0)
        df['range'] = df['high'] - df['low']
        df['body'] = abs(df['close'] - df['open'])
        df['direction'] = np.where(df['close'] > df['open'], 1, -1)
        df['order_flow'] = df['volume'] * df['direction']
        
        # Technical indicators
        df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
        df['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Volatility measures
        df['volatility'] = df['range'].rolling(self.config['volatility_lookback']).mean()
        df['true_range'] = talib.TRANGE(df['high'], df['low'], df['close'])
        df['volatility_ratio'] = df['true_range'] / df['close'].rolling(20).std()
        
        # Market regime features
        df['regime_type'] = np.where(
            (df['close'] > df['ema_200']) & (df['adx_14'] > 25), 
            'uptrend',
            np.where(
                (df['close'] < df['ema_200']) & (df['adx_14'] > 25),
                'downtrend',
                'range'
            )
        )
        
        # Create target variables (future price movements)
        horizon = self.config['prediction_horizon']
        df['future_return'] = df['close'].pct_change(horizon).shift(-horizon)
        df['future_direction'] = np.where(df['future_return'] > 0, 1, 0)
        df['future_high'] = df['high'].rolling(horizon).max().shift(-horizon)
        df['future_low'] = df['low'].rolling(horizon).min().shift(-horizon)
        
        # Drop NA values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        return df

    def learn_market_dynamics(self):
        print("\nLearning market dynamics from historical data...")
        
        for symbol in self.config['symbols']:
            print(f"\nAnalyzing {symbol}...")
            
            if symbol not in self.symbol_info or not self.symbol_info[symbol]['trade_allowed']:
                print(f"  [!] Symbol {symbol} not available or not tradable, skipping")
                continue
                
            self.market_models[symbol] = {}
            self.market_knowledge[symbol] = {}
            
            for timeframe in self.config['timeframes']:
                print(f"\nLearning market behavior for {symbol} on timeframe {timeframe}...")
                
                start_date = datetime.now() - timedelta(days=self.config['learning_period'])
                df = self.get_historical_data(symbol, timeframe, start_date)
                
                if df is None:
                    print(f"  [!] No historical data available for {symbol} on {timeframe}")
                    continue
                
                # Train machine learning model
                self._train_market_model(symbol, timeframe, df)
                
                # Extract market behaviors
                self._extract_market_behaviors(symbol, timeframe, df)

    def _train_market_model(self, symbol, timeframe, df):
        print(f"Training predictive model for {symbol} on {timeframe}...")
        
        # Prepare features and targets
        feature_cols = ['returns', 'range', 'body', 'direction', 'order_flow', 
                       'rsi_14', 'atr_14', 'adx_14', 'ema_20', 'ema_50', 'ema_200',
                       'macd', 'macd_signal', 'obv', 'volatility', 'true_range',
                       'volatility_ratio']
        
        X = df[feature_cols]
        y_return = df['future_return']
        y_direction = df['future_direction']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_direction, test_size=0.2, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with improved parameters
        model = RandomForestRegressor(
            n_estimators=200,  # Increased number of trees
            max_depth=12,     # Slightly deeper trees
            min_samples_split=3,  # More flexible splitting
            min_samples_leaf=2,   # Smaller leaf size
            max_features='sqrt',  # Better generalization
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model with additional metrics
        train_preds = model.predict(X_train_scaled)
        test_preds = model.predict(X_test_scaled)
        
        # Convert probabilities to binary predictions for classification metrics
        train_preds_binary = np.where(train_preds > 0.5, 1, 0)
        test_preds_binary = np.where(test_preds > 0.5, 1, 0)
        
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        train_accuracy = accuracy_score(y_train, train_preds_binary)
        test_accuracy = accuracy_score(y_test, test_preds_binary)
        train_precision = precision_score(y_train, train_preds_binary)
        test_precision = precision_score(y_test, test_preds_binary)
        
        print(f"Model trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
        print(f"Train Accuracy: {train_accuracy:.3f}, Test Accuracy: {test_accuracy:.3f}")
        print(f"Train Precision: {train_precision:.3f}, Test Precision: {test_precision:.3f}")
        
        # Enhanced confidence calculation
        train_confidence = self._calculate_model_confidence(train_preds, y_train)
        test_confidence = self._calculate_model_confidence(test_preds, y_test)
        
        # Store model and scaler
        self.market_models[symbol][timeframe] = {
            'model': model,
            'scaler': scaler,
            'feature_cols': feature_cols,
            'last_trained': datetime.now(),
            'train_score': train_score,
            'test_score': test_score,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_confidence': train_confidence,
            'test_confidence': test_confidence
        }

    def _calculate_model_confidence(self, predictions, true_values):
        """Enhanced confidence metric calculation with additional checks"""
        # Calculate prediction errors
        errors = np.abs(predictions - true_values)
        
        # Calculate normalized confidence scores
        confidence_scores = 1 - (errors / np.max(errors))
        
        # Weight by prediction magnitude (strong predictions get higher confidence)
        weighted_scores = confidence_scores * np.abs(predictions - 0.5)
        
        # Calculate overall confidence metric
        confidence_metric = np.mean(weighted_scores) * 2  # Scale to 0-1 range
        
        # Apply sigmoid function to smooth the confidence score
        confidence_metric = 1 / (1 + np.exp(-10 * (confidence_metric - 0.5)))
        
        # Additional check for prediction consistency
        directional_consistency = np.mean(np.sign(predictions - 0.5) == np.sign(np.mean(predictions) - 0.5))
        confidence_metric *= directional_consistency
        
        return min(max(confidence_metric, 0), 1)  # Ensure within [0,1] range

    def _extract_market_behaviors(self, symbol, timeframe, df):
        print(f"Extracting market behaviors for {symbol} on {timeframe}...")
        
        behaviors = {
            'volatility_regimes': self._analyze_volatility_regimes(df),
            'time_of_day_effects': self._analyze_time_of_day_effects(df),
            'price_action_tendencies': self._analyze_price_action_tendencies(df),
            'support_resistance_levels': self._identify_support_resistance(df),
            'regime_transitions': self._analyze_regime_transitions(df),
            'last_updated': datetime.now()
        }
        
        self.market_knowledge[symbol][timeframe] = behaviors
        
        # Print summary of learned behaviors
        print(f"\nLearned behaviors for {symbol} on {timeframe}:")
        print(f"- Volatility regimes: {len(behaviors['volatility_regimes'])} identified")
        print(f"- Time of day effects: {len(behaviors['time_of_day_effects'])} patterns")
        print(f"- Price action tendencies: {len(behaviors['price_action_tendencies'])} patterns")
        print(f"- Key levels: {len(behaviors['support_resistance_levels'])} S/R levels")
        print(f"- Regime transitions: {len(behaviors['regime_transitions'])} patterns")

    def _analyze_volatility_regimes(self, df):
        """Identify different volatility regimes in the market"""
        volatility = df['volatility']
        regimes = []
        
        # Define volatility thresholds
        low_threshold = volatility.quantile(0.25)
        high_threshold = volatility.quantile(0.75)
        
        current_regime = None
        start_idx = 0
        
        for i in range(len(volatility)):
            if volatility.iloc[i] < low_threshold:
                regime = 'low'
            elif volatility.iloc[i] > high_threshold:
                regime = 'high'
            else:
                regime = 'medium'
            
            if regime != current_regime:
                if current_regime is not None:
                    # Save previous regime
                    regimes.append({
                        'type': current_regime,
                        'start': df.index[start_idx],
                        'end': df.index[i-1],
                        'duration': (df.index[i-1] - df.index[start_idx]).total_seconds() / 3600,
                        'avg_volatility': volatility.iloc[start_idx:i].mean(),
                        'price_change': df['close'].iloc[i-1] - df['close'].iloc[start_idx]
                    })
                current_regime = regime
                start_idx = i
        
        # Add the last regime
        if current_regime is not None:
            regimes.append({
                'type': current_regime,
                'start': df.index[start_idx],
                'end': df.index[-1],
                'duration': (df.index[-1] - df.index[start_idx]).total_seconds() / 3600,
                'avg_volatility': volatility.iloc[start_idx:].mean(),
                'price_change': df['close'].iloc[-1] - df['close'].iloc[start_idx]
            })
        
        return regimes

    def _analyze_time_of_day_effects(self, df):
        """Analyze how price behaves at different times of day"""
        df['hour'] = df.index.hour
        hour_groups = df.groupby('hour')
        
        effects = []
        
        for hour, group in hour_groups:
            avg_return = group['returns'].mean()
            pos_percentage = (group['direction'] > 0).mean()
            avg_volatility = group['volatility'].mean()
            
            effects.append({
                'hour': hour,
                'avg_return': avg_return,
                'pos_percentage': pos_percentage,
                'avg_volatility': avg_volatility,
                'sample_size': len(group)
            })
        
        return sorted(effects, key=lambda x: x['avg_return'], reverse=True)

    def _analyze_price_action_tendencies(self, df):
        """Identify recurring price action patterns and their outcomes"""
        tendencies = []
        
        # Analyze candle sequences
        for i in range(2, len(df)-1):
            prev_candle = df.iloc[i-1]
            current_candle = df.iloc[i]
            next_candle = df.iloc[i+1]
            
            # Engulfing patterns
            if (prev_candle['direction'] == -1 and current_candle['direction'] == 1 and
                current_candle['body'] > prev_candle['body'] and
                current_candle['close'] > prev_candle['open'] and
                current_candle['open'] < prev_candle['close']):
                
                outcome = next_candle['close'] - current_candle['close']
                tendencies.append({
                    'type': 'bullish_engulfing',
                    'outcome': outcome,
                    'rsi': current_candle['rsi_14'],
                    'volatility': current_candle['volatility'],
                    'regime': current_candle['regime_type']
                })
            
            elif (prev_candle['direction'] == 1 and current_candle['direction'] == -1 and
                  current_candle['body'] > prev_candle['body'] and
                  current_candle['close'] < prev_candle['open'] and
                  current_candle['open'] > prev_candle['close']):
                
                outcome = next_candle['close'] - current_candle['close']
                tendencies.append({
                    'type': 'bearish_engulfing',
                    'outcome': outcome,
                    'rsi': current_candle['rsi_14'],
                    'volatility': current_candle['volatility'],
                    'regime': current_candle['regime_type']
                })
        
        # Group similar tendencies
        grouped_tendencies = {}
        for tendency in tendencies:
            key = (tendency['type'], tendency['regime'], 
                  tendency['rsi'] // 10, tendency['volatility'] // (df['volatility'].mean() / 5))
            
            if key not in grouped_tendencies:
                grouped_tendencies[key] = {
                    'type': tendency['type'],
                    'regime': tendency['regime'],
                    'rsi_range': (key[2]*10, (key[2]+1)*10),
                    'volatility_range': (key[3]*(df['volatility'].mean()/5), (key[3]+1)*(df['volatility'].mean()/5)),
                    'outcomes': [],
                    'count': 0
                }
            
            grouped_tendencies[key]['outcomes'].append(tendency['outcome'])
            grouped_tendencies[key]['count'] += 1
        
        # Calculate statistics for each group
        results = []
        for key, group in grouped_tendencies.items():
            if group['count'] < 5:  # Minimum sample size
                continue
                
            outcomes = np.array(group['outcomes'])
            win_rate = (outcomes > 0).mean()
            avg_outcome = outcomes.mean()
            expectancy = avg_outcome * win_rate - outcomes[outcomes <= 0].mean() * (1 - win_rate)
            
            results.append({
                'pattern': group['type'],
                'regime': group['regime'],
                'rsi_range': group['rsi_range'],
                'volatility_range': group['volatility_range'],
                'win_rate': win_rate,
                'avg_outcome': avg_outcome,
                'expectancy': expectancy,
                'count': group['count']
            })
        
        return sorted(results, key=lambda x: x['expectancy'], reverse=True)

    def _identify_support_resistance(self, df, lookback=100):
        """Identify key support and resistance levels"""
        levels = []
        price_values = df['close'].values
        
        for i in range(lookback, len(price_values)-lookback):
            window = price_values[i-lookback:i+lookback]
            current_price = price_values[i]
            
            # Check for support
            if current_price == np.min(window):
                levels.append({
                    'price': current_price,
                    'type': 'support',
                    'time': df.index[i],
                    'strength': 1  # Will be updated based on touches
                })
            
            # Check for resistance
            elif current_price == np.max(window):
                levels.append({
                    'price': current_price,
                    'type': 'resistance',
                    'time': df.index[i],
                    'strength': 1
                })
        
        # Merge nearby levels and count touches
        merged_levels = []
        tolerance = df['atr_14'].mean() * 0.5
        
        for level in sorted(levels, key=lambda x: x['price']):
            found = False
            for merged in merged_levels:
                if abs(level['price'] - merged['price']) <= tolerance and level['type'] == merged['type']:
                    merged['strength'] += 1
                    merged['time'] = max(merged['time'], level['time'])  # Keep most recent
                    found = True
                    break
            
            if not found:
                merged_levels.append(level)
        
        # Filter by strength and recency
        recent_levels = [l for l in merged_levels 
                        if (datetime.now() - l['time']).days < 30 and l['strength'] >= 2]
        
        return sorted(recent_levels, key=lambda x: x['strength'], reverse=True)

    def _analyze_regime_transitions(self, df):
        """Analyze how the market transitions between different regimes"""
        transitions = []
        regimes = df['regime_type'].values
        
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                transition = {
                    'from': regimes[i-1],
                    'to': regimes[i],
                    'time': df.index[i],
                    'rsi': df['rsi_14'].iloc[i],
                    'volatility': df['volatility'].iloc[i],
                    'price_change_next_5': df['close'].iloc[i+5] - df['close'].iloc[i] if i+5 < len(df) else None
                }
                transitions.append(transition)
        
        # Group similar transitions
        grouped_transitions = defaultdict(list)
        for t in transitions:
            key = (t['from'], t['to'], 
                  int(t['rsi'] // 10), 
                  int(t['volatility'] // (df['volatility'].mean() / 5)))
            grouped_transitions[key].append(t)
        
        # Calculate transition statistics
        results = []
        for key, group in grouped_transitions.items():
            if len(group) < 3:  # Minimum sample size
                continue
                
            price_changes = [t['price_change_next_5'] for t in group if t['price_change_next_5'] is not None]
            if not price_changes:
                continue
                
            avg_change = np.mean(price_changes)
            pos_percentage = (np.array(price_changes) > 0).mean()
            
            results.append({
                'from': key[0],
                'to': key[1],
                'rsi_range': (key[2]*10, (key[2]+1)*10),
                'volatility_range': (key[3]*(df['volatility'].mean()/5), (key[3]+1)*(df['volatility'].mean()/5)),
                'avg_price_change': avg_change,
                'pos_percentage': pos_percentage,
                'count': len(group)
            })
        
        return sorted(results, key=lambda x: abs(x['avg_price_change']), reverse=True)

    def analyze_current_market(self, symbol, timeframe):
        """Analyze current market conditions and make predictions"""
        if symbol not in self.market_models or timeframe not in self.market_models[symbol]:
            print(f"No model available for {symbol} on {timeframe}")
            return None
            
        # Get recent data
        df = self.get_historical_data(symbol, timeframe, datetime.now() - timedelta(days=30))
        if df is None:
            print(f"Could not get recent data for {symbol} on {timeframe}")
            return None
        
        # Prepare current features
        current_features = df[self.market_models[symbol][timeframe]['feature_cols']].iloc[-1].values.reshape(1, -1)
        scaler = self.market_models[symbol][timeframe]['scaler']
        scaled_features = scaler.transform(current_features)
        
        # Make prediction
        model = self.market_models[symbol][timeframe]['model']
        prediction = model.predict(scaled_features)[0]
        
        # Calculate enhanced confidence score
        recent_data = df.iloc[-30:]  # Use last 30 periods for confidence calculation
        recent_features = recent_data[self.market_models[symbol][timeframe]['feature_cols']]
        scaled_recent_features = scaler.transform(recent_features)
        recent_predictions = model.predict(scaled_recent_features)
        
        # Calculate volatility-adjusted confidence
        prediction_std = np.std(recent_predictions)
        volatility_factor = 1 / (1 + prediction_std)  # Higher volatility reduces confidence
        
        # Calculate directional consistency
        directional_consistency = np.mean(np.sign(recent_predictions - 0.5) == np.sign(prediction - 0.5))
        
        # Final confidence calculation
        base_confidence = abs(prediction - 0.5) * 2  # Original 0-1 scale
        enhanced_confidence = base_confidence * volatility_factor * directional_consistency
        
        # Get current market state
        current_state = {
            'price': df['close'].iloc[-1],
            'rsi': df['rsi_14'].iloc[-1],
            'volatility': df['volatility'].iloc[-1],
            'regime': df['regime_type'].iloc[-1],
            'time_of_day': df.index[-1].hour,
            'near_support': self._is_near_level(symbol, timeframe, df['close'].iloc[-1], 'support'),
            'near_resistance': self._is_near_level(symbol, timeframe, df['close'].iloc[-1], 'resistance'),
            'predicted_direction': prediction,
            'confidence': enhanced_confidence,
            'volatility_factor': volatility_factor,
            'directional_consistency': directional_consistency
        }
        
        # Find similar historical scenarios
        similar_scenarios = self._find_similar_scenarios(symbol, timeframe, current_state)
        
        analysis = {
            'current_state': current_state,
            'prediction': prediction,
            'similar_scenarios': similar_scenarios,
            'timestamp': datetime.now()
        }
        
        return analysis

    def _is_near_level(self, symbol, timeframe, price, level_type):
        """Check if price is near a known support/resistance level"""
        if symbol not in self.market_knowledge or timeframe not in self.market_knowledge[symbol]:
            return False
            
        levels = self.market_knowledge[symbol][timeframe]['support_resistance_levels']
        relevant_levels = [l for l in levels if l['type'] == level_type]
        
        if not relevant_levels:
            return False
            
        closest_level = min(relevant_levels, key=lambda x: abs(x['price'] - price))
        atr = self._get_current_atr(symbol, timeframe)
        
        return abs(closest_level['price'] - price) <= atr * 1.5

    def _get_current_atr(self, symbol, timeframe):
        """Get current ATR value with more robust validation"""
        # First try to get recent data
        df = self.get_historical_data(symbol, timeframe, datetime.now() - timedelta(days=15))
        
        if df is None or len(df) < 14:  # Need at least 14 periods for ATR
            print(f"Warning: Insufficient data for {symbol} on {timeframe} - using price-based fallback")
            current_price = mt5.symbol_info_tick(symbol).ask
            return current_price * 0.01  # 1% of price as fallback
        
        try:
            # Try to get pre-calculated ATR first
            if 'atr_14' in df.columns:
                atr = df['atr_14'].iloc[-1]
                if not pd.isna(atr) and atr > 0:
                    return atr
            
            # Calculate ATR manually if needed
            high = df['high'].iloc[-14:].values
            low = df['low'].iloc[-14:].values
            close = df['close'].iloc[-14:].values
            
            if len(high) < 14 or len(low) < 14 or len(close) < 14:
                raise ValueError("Not enough data points for ATR calculation")
                
            atr = talib.ATR(high, low, close, timeperiod=14)[-1]
            
            if pd.isna(atr) or atr <= 0:
                raise ValueError("Invalid ATR value calculated")
                
            return atr
            
        except Exception as e:
            print(f"ATR calculation failed for {symbol}: {str(e)}")
            current_price = df['close'].iloc[-1] if len(df) > 0 else mt5.symbol_info_tick(symbol).ask
            return current_price * 0.01  # 1% of price as final fallback

    def _find_similar_scenarios(self, symbol, timeframe, current_state):
        """Find historical scenarios similar to current market conditions"""
        similar_scenarios = []
        
        # Get historical data
        df = self.get_historical_data(symbol, timeframe, datetime.now() - timedelta(days=self.config['learning_period']))
        if df is None:
            return similar_scenarios
        
        # Define similarity criteria
        rsi_tolerance = 5
        vol_tolerance = df['volatility'].mean() * 0.3
        regime = current_state['regime']
        time_of_day = current_state['time_of_day']
        
        # Find similar historical periods
        for i in range(len(df)):
            if (df['regime_type'].iloc[i] == regime and
                abs(df['rsi_14'].iloc[i] - current_state['rsi']) <= rsi_tolerance and
                abs(df['volatility'].iloc[i] - current_state['volatility']) <= vol_tolerance and
                df.index[i].hour == time_of_day):
                
                # Check if near similar levels
                near_support = self._is_near_level(symbol, timeframe, df['close'].iloc[i], 'support')
                near_resistance = self._is_near_level(symbol, timeframe, df['close'].iloc[i], 'resistance')
                
                if near_support != current_state['near_support'] or near_resistance != current_state['near_resistance']:
                    continue
                
                # Get outcome
                if i + self.config['prediction_horizon'] < len(df):
                    outcome = df['close'].iloc[i + self.config['prediction_horizon']] - df['close'].iloc[i]
                    scenario = {
                        'time': df.index[i],
                        'outcome': outcome,
                        'rsi': df['rsi_14'].iloc[i],
                        'volatility': df['volatility'].iloc[i],
                        'price': df['close'].iloc[i],
                        'regime': df['regime_type'].iloc[i]
                    }
                    similar_scenarios.append(scenario)
        
        # Calculate statistics
        if not similar_scenarios:
            print(f"No similar historical scenarios found for current market conditions")
            return None
            
        outcomes = np.array([s['outcome'] for s in similar_scenarios])
        avg_outcome = outcomes.mean()
        win_rate = (outcomes > 0).mean()
        expectancy = avg_outcome * win_rate - outcomes[outcomes <= 0].mean() * (1 - win_rate)
        
        return {
            'count': len(similar_scenarios),
            'avg_outcome': avg_outcome,
            'win_rate': win_rate,
            'expectancy': expectancy,
            'scenarios': similar_scenarios[:10]  # Return first 10 examples
        }

    def calculate_position_size(self, symbol, stop_loss_pips):
        if symbol not in self.symbol_info:
            print(f"Symbol {symbol} not found in symbol info")
            return 0
            
        # Validate stop_loss_pips
        if stop_loss_pips <= 0:
            print(f"Invalid stop loss pips: {stop_loss_pips}")
            return 0
            
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info")
            return 0
            
        balance = account_info.balance
        if balance <= 0:
            print(f"Invalid account balance: {balance}")
            return 0
            
        risk_amount = balance * (self.config['risk_percent'] / 100)
        pip_value = self._calculate_pip_value(symbol)
        
        if pip_value <= 0:
            print(f"Invalid pip value: {pip_value}")
            return 0
            
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Round to nearest lot step with validation
        symbol_data = self.symbol_info[symbol]
        
        # Ensure position size is at least the minimum lot size
        if position_size < symbol_data['min_lot']:
            print(f"Calculated size {position_size} below minimum {symbol_data['min_lot']}, using minimum")
            position_size = symbol_data['min_lot']
        
        position_size = max(symbol_data['min_lot'], min(symbol_data['max_lot'], position_size))
        
        if symbol_data['lot_step'] > 0:
            position_size = round(position_size / symbol_data['lot_step']) * symbol_data['lot_step']
        else:
            position_size = round(position_size, 2)
        
        print(f"Calculated position size: {position_size} lots for {symbol} with {stop_loss_pips} pip SL")
        return position_size

    def _calculate_pip_value(self, symbol):
        if symbol not in self.symbol_info:
            return 0
        return self.symbol_info[symbol]['pip_value'] * self.symbol_info[symbol]['contract_size']

    def _price_to_pips(self, symbol, price_diff):
        if symbol not in self.symbol_info:
            return 0
        return price_diff / self.symbol_info[symbol]['pip_value']

    def _pips_to_price(self, symbol, pips):
        if symbol not in self.symbol_info:
            return 0
        return pips * self.symbol_info[symbol]['pip_value']

    def execute_trade(self, symbol, direction, analysis):
        try:
            # Check for existing trades
            if symbol in self.active_trades:
                print(f"Already have an active trade for {symbol}, skipping new trade")
                return False
                
            # Get tick data with error handling
            try:
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    print(f"Failed to get tick data for {symbol}")
                    return False
            except Exception as e:
                print(f"Error getting tick data for {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
                
            current_price = tick.ask if direction > 0 else tick.bid
            
            # Validate symbol info exists
            if symbol not in self.symbol_info:
                print(f"Symbol info not found for {symbol}")
                return False
                
            spread_pips = self.symbol_info[symbol].get('current_spread_pips', 0)
            
            print(f"\nPreparing to execute {'BUY' if direction > 0 else 'SELL'} trade for {symbol}")
            print(f"Current price: {current_price}")
            print(f"Current spread: {spread_pips:.1f} pips")
            
            # Get current volatility with robust error handling
            try:
                current_atr = self._get_current_atr(symbol, analysis['timeframe'])
                if current_atr <= 0:
                    current_atr = current_price * 0.01  # Fallback to 1% of price
                    print(f"Warning: Invalid ATR, using fallback volatility value: {current_atr:.5f}")
                else:
                    print(f"Current ATR: {current_atr:.5f}")
            except Exception as e:
                print(f"Error calculating ATR for {symbol}: {str(e)}")
                current_atr = current_price * 0.01  # Fallback
                import traceback
                traceback.print_exc()
            
            # Calculate stop distance with validation
            try:
                min_stop_distance = max(
                    self.symbol_info[symbol].get('min_stop_distance', 0),
                    current_atr * 0.5  # At least half ATR
                )
            except Exception as e:
                print(f"Error calculating stop distance for {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
            
            # Calculate stop loss with comprehensive error handling
            try:
                if direction > 0:
                    if analysis['current_state'].get('near_support', False):
                        support_levels = [l for l in self.market_knowledge[symbol][analysis['timeframe']]['support_resistance_levels'] 
                                        if l['type'] == 'support']
                        if support_levels:
                            closest_support = min(support_levels, key=lambda x: abs(x['price'] - current_price))
                            sl_price = closest_support['price'] - min_stop_distance
                            print(f"Placing SL below support level at {closest_support['price']}")
                        else:
                            sl_price = current_price - min_stop_distance * 1.5
                            print(f"No support levels found, using volatility-based SL")
                    else:
                        sl_price = current_price - min_stop_distance * 1.5
                        print(f"Placing SL at {min_stop_distance * 1.5:.5f} below price (volatility-based)")
                else:
                    if analysis['current_state'].get('near_resistance', False):
                        resistance_levels = [l for l in self.market_knowledge[symbol][analysis['timeframe']]['support_resistance_levels'] 
                                        if l['type'] == 'resistance']
                        if resistance_levels:
                            closest_resistance = min(resistance_levels, key=lambda x: abs(x['price'] - current_price))
                            sl_price = closest_resistance['price'] + min_stop_distance
                            print(f"Placing SL above resistance level at {closest_resistance['price']}")
                        else:
                            sl_price = current_price + min_stop_distance * 1.5
                            print(f"No resistance levels found, using volatility-based SL")
                    else:
                        sl_price = current_price + min_stop_distance * 1.5
                        print(f"Placing SL at {min_stop_distance * 1.5:.5f} above price (volatility-based)")
            except Exception as e:
                print(f"Error calculating stop loss for {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
            
            # Convert to pips with error handling
            try:
                stop_loss_pips = self._price_to_pips(symbol, abs(current_price - sl_price))
                if stop_loss_pips <= 0:
                    stop_loss_pips = self.symbol_info[symbol].get('min_stop_pips', 10) * 1.5
                    print(f"Warning: Invalid stop loss pips, using minimum stop distance: {stop_loss_pips:.1f} pips")
            except Exception as e:
                print(f"Error converting stop loss to pips for {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
            
            # Calculate take profit
            try:
                take_profit_pips = stop_loss_pips * self.config['risk_reward_ratio']
                print(f"Take profit distance: {take_profit_pips:.1f} pips (RR: {self.config['risk_reward_ratio']}:1)")
            except Exception as e:
                print(f"Error calculating take profit for {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
            
            # Calculate position size with validation
            try:
                position_size = self.calculate_position_size(symbol, stop_loss_pips)
                if position_size <= 0:
                    print(f"Invalid position size calculated: {position_size}")
                    return False
            except Exception as e:
                print(f"Error calculating position size for {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
            
            # Prepare entry prices
            try:
                if direction > 0:
                    entry_price = tick.ask
                    tp_price = entry_price + self._pips_to_price(symbol, take_profit_pips)
                    sl_price = entry_price - self._pips_to_price(symbol, stop_loss_pips)
                else:
                    entry_price = tick.bid
                    tp_price = entry_price - self._pips_to_price(symbol, take_profit_pips)
                    sl_price = entry_price + self._pips_to_price(symbol, stop_loss_pips)

                # === Round all price levels based on symbol's digits ===
                symbol_info = mt5.symbol_info(symbol)
                if not symbol_info:
                    print(f"Error: Could not fetch symbol_info for {symbol}")
                    return False

                digits = symbol_info.digits
                point = symbol_info.point
                stops_level = symbol_info.trade_stops_level * point

                entry_price = round(entry_price, digits)
                sl_price = round(sl_price, digits)
                tp_price = round(tp_price, digits)

                # === Validate stop level distances ===
                if abs(entry_price - sl_price) < stops_level:
                    print(f"SL too close to price for {symbol}: {abs(entry_price - sl_price)} < min {stops_level}")
                    return False

                if abs(tp_price - entry_price) < stops_level:
                    print(f"TP too close to price for {symbol}: {abs(tp_price - entry_price)} < min {stops_level}")
                    return False

                print(f"Rounded Entry Price: {entry_price}")
                print(f"Rounded SL Price: {sl_price}")
                print(f"Rounded TP Price: {tp_price}")
                print(f"Minimum stop level for {symbol}: {stops_level}")

            except Exception as e:
                print(f"Error preparing and validating price levels for {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False

            # Prepare trade request with margin check
            try:
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": position_size,
                    "type": mt5.ORDER_TYPE_BUY if direction > 0 else mt5.ORDER_TYPE_SELL,
                    "price": entry_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "deviation": 10,
                    "magic": 123456,
                    "comment": f"Conf:{round(analysis['current_state'].get('confidence', 0) * 100, 1)}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                print("\nOrder Request to Check:")
                for k, v in request.items():
                    print(f"  {k}: {v}")

                check_result = mt5.order_check(request)
                if check_result is None:
                    print(f"Failed to check order requirements for {symbol}")
                    
                    # Additional diagnostic info
                    last_error = mt5.last_error()
                    print(f"MT5 last error: {last_error}")

                    # Try enabling the symbol if it's not visible
                    if not mt5.symbol_info(symbol).visible:
                        print(f"{symbol} not visible. Attempting to enable it.")
                        if not mt5.symbol_select(symbol, True):
                            print(f"Failed to enable {symbol}")
                                        
                    return False

                print("🔍 order_check() result details:")
                for attr in dir(check_result):
                    if not attr.startswith("_") and not callable(getattr(check_result, attr)):
                        print(f"    {attr}: {getattr(check_result, attr)}")

                if check_result.retcode != mt5.TRADE_RETCODE_DONE and check_result.retcode != 0:
                    print(f"[❌] Order check failed for {symbol}")
                    print(f"  Retcode: {check_result.retcode}")
                    print(f"  Comment: {check_result.comment}")
                    return False
                else:
                    print(f"[✅] Order check passed for {symbol}")

            except Exception as e:
                print(f"Error preparing trade request for {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False

            
            # Execute trade with comprehensive error handling
            try:
                result = mt5.order_send(request)
                
                if result is None:
                    print(f"Trade execution returned None for {symbol}")
                    return False
                    
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"\nSuccessfully entered {'BUY' if direction > 0 else 'SELL'} trade for {symbol}")
                    print(f"  Size: {position_size} lots")
                    print(f"  Entry: {entry_price}")
                    print(f"  TP: {tp_price} ({take_profit_pips:.1f} pips)")
                    print(f"  SL: {sl_price} ({stop_loss_pips:.1f} pips)")
                    print(f"  Model confidence: {analysis['current_state'].get('confidence', 0):.2%}")
                    print(f"  Volatility factor: {analysis['current_state'].get('volatility_factor', 0):.2f}")
                    print(f"  Directional consistency: {analysis['current_state'].get('directional_consistency', 0):.2%}")
                    
                    if analysis.get('similar_scenarios'):
                        print(f"  Historical win rate: {analysis['similar_scenarios'].get('win_rate', 0):.2%}")
                        print(f"  Historical expectancy: {analysis['similar_scenarios'].get('expectancy', 0):.6f}")
                    
                    self.active_trades[symbol] = {
                        'ticket': result.order,
                        'direction': direction,
                        'entry': entry_price,
                        'size': position_size,
                        'tp': tp_price,
                        'sl': sl_price,
                        'analysis': analysis,
                        'open_time': datetime.now(),
                        'highest_price': entry_price if direction > 0 else entry_price,
                        'lowest_price': entry_price if direction < 0 else entry_price,
                        'atr': current_atr,
                        'trailing_active': False  # Flag to track if trailing is active
                    }
                    return True
                else:
                    print(f"Trade failed for {symbol}, error code: {result.retcode}")
                    if hasattr(result, 'comment'):
                        print(f"Error details: {result.comment}")
                    return False
                    
            except Exception as e:
                print(f"Error executing trade for {symbol}: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
                
        except Exception as e:
            print(f"Unexpected error in execute_trade for {symbol}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def update_trailing_stops(self):
        """Update trailing stops based on current price action"""
        if not self.active_trades:
            return
            
        positions = mt5.positions_get()
        if positions is None:
            print("Failed to get open positions for trailing stop update")
            return
            
        for symbol, trade in list(self.active_trades.items()):
            position = next((p for p in positions if p.symbol == symbol and p.ticket == trade['ticket']), None)
            if not position:
                print(f"Trade {trade['ticket']} for {symbol} not found in open positions, removing from active trades")
                self.active_trades.pop(symbol, None)
                continue
                
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                print(f"Failed to get tick data for {symbol}")
                continue
                
            current_price = tick.ask if trade['direction'] > 0 else tick.bid
            
            # Update highest/lowest prices
            if trade['direction'] > 0:
                trade['highest_price'] = max(trade['highest_price'], current_price)
                price_change = current_price - trade['entry']
            else:
                trade['lowest_price'] = min(trade['lowest_price'], current_price)
                price_change = trade['entry'] - current_price
            
            # Calculate dynamic trailing distance (between min and max in ATR multiples)
            trail_distance = trade['atr'] * (
                self.config['min_trail_distance'] + 
                (self.config['max_trail_distance'] - self.config['min_trail_distance']) * 
                self.config['adaptive_trailing_factor']
            )
            
            # Only activate trailing after price has moved in our favor by at least 1 ATR
            if not trade.get('trailing_active', False):
                required_move = trade['atr'] * 1.0  # 1 ATR in our favor
                if ((trade['direction'] > 0 and price_change >= required_move) or
                    (trade['direction'] < 0 and price_change >= required_move)):
                    trade['trailing_active'] = True
                    print(f"Trailing stop activated for {symbol}")
            
            # Only update if trailing is active
            if trade.get('trailing_active', False):
                # Calculate new stop level
                if trade['direction'] > 0:
                    new_sl = trade['highest_price'] - trail_distance
                    # Ensure new stop is above entry and previous stop
                    if new_sl > max(trade['sl'], trade['entry'] - trail_distance):
                        trade['sl'] = new_sl
                        print(f"New trailing stop for {symbol} long: {new_sl:.5f} (distance: {trail_distance:.5f})")
                else:
                    new_sl = trade['lowest_price'] + trail_distance
                    # Ensure new stop is below entry and previous stop
                    if new_sl < min(trade['sl'], trade['entry'] + trail_distance):
                        trade['sl'] = new_sl
                        print(f"New trailing stop for {symbol} short: {new_sl:.5f} (distance: {trail_distance:.5f})")
                
                # Update stop loss if changed
                if ((trade['direction'] > 0 and position.sl < trade['sl']) or 
                    (trade['direction'] < 0 and position.sl > trade['sl'])):
                    
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "sl": trade['sl'],
                        "tp": trade['tp'],
                        "position": trade['ticket'],
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        print(f"Successfully updated trailing stop for {symbol} to {trade['sl']}")
                    else:
                        print(f"Failed to update trailing stop for {symbol}, error: {result.retcode}")

    def check_open_trades(self):
        if not self.active_trades:
            print("No active trades to check")
            return
            
        positions = mt5.positions_get()
        if positions is None:
            print("Failed to get open positions")
            return
            
        current_profit = 0
        trades_to_close = []
        
        for symbol, trade in list(self.active_trades.items()):
            position = next((p for p in positions if p.symbol == symbol and p.ticket == trade['ticket']), None)
            
            if not position:
                print(f"Trade {trade['ticket']} for {symbol} not found in open positions")
                self.active_trades.pop(symbol, None)
                continue
                
            current_profit += position.profit
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                continue
                
            current_price = tick.ask if trade['direction'] > 0 else tick.bid
            
            duration = (datetime.now() - trade['open_time']).total_seconds() / 60
            print(f"\n{symbol} trade status:")
            print(f"  Duration: {duration:.1f} minutes")
            print(f"  Current price: {current_price}")
            print(f"  Entry: {trade['entry']}")
            print(f"  Current profit: ${position.profit:.2f}")
            print(f"  TP: {trade['tp']}")
            print(f"  SL: {trade['sl']}")
            
            if (trade['direction'] > 0 and current_price >= trade['tp']) or \
               (trade['direction'] < 0 and current_price <= trade['tp']):
                print(f"  TP HIT for {symbol}!")
                trades_to_close.append(position.ticket)
            elif (trade['direction'] > 0 and current_price <= trade['sl']) or \
                 (trade['direction'] < 0 and current_price >= trade['sl']):
                print(f"  SL HIT for {symbol}!")
                trades_to_close.append(position.ticket)
        
        # Calculate cumulative profit from closed positions
        account_info = mt5.account_info()
        if account_info is not None:
            current_balance = account_info.balance
            self.total_profit = current_balance - account_info.equity  # Track realized profit
            
            print(f"\nTotal realized profit: ${self.total_profit:.2f}")
            
            if not self.target_hit and self.total_profit >= self.config['target_profit_usd']:
                print(f"\nTARGET PROFIT OF ${self.config['target_profit_usd']} HIT!")
                self.target_hit = True
                self.last_target_hit_time = datetime.now()
                trades_to_close.extend([p.ticket for p in positions])
        
        for ticket in trades_to_close:
            self.close_trade(ticket)

    def close_trade(self, ticket):
        position = mt5.positions_get(ticket=ticket)
        if not position:
            print(f"No position found with ticket {ticket}")
            return False
            
        position = position[0]
        symbol = position.symbol
        volume = position.volume
        position_type = position.type
        
        if position_type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 10,
            "magic": 123456,
            "comment": "Close trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Successfully closed trade {ticket} for {symbol}")
            if symbol in self.active_trades:
                self.active_trades.pop(symbol)
            return True
        else:
            print(f"Failed to close trade {ticket}, error: {result.retcode}")
            return False

    def close_all_trades(self):
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            print("No open positions to close")
            return
            
        print(f"Closing all {len(positions)} open positions...")
        for position in positions:
            self.close_trade(position.ticket)

    def can_trade(self):
        if self.target_hit:
            if (datetime.now() - self.last_target_hit_time) > timedelta(hours=24):
                print("24 hours have passed since target was hit, resetting target")
                self.target_hit = False
                self.total_profit = 0
            else:
                print("Target profit already hit today, waiting for 24 hours to pass")
                return False
                
        return True

    def run(self):
        print("\nStarting Market Understanding Bot...")
        last_learning_update = datetime.now()
        
        try:
            while True:
                try:
                    print("\n" + "="*50)
                    print(f"Cycle started at {datetime.now()}")
                    
                    self.check_open_trades()
                    self.update_trailing_stops()
                    
                    # Periodic model retraining
                    if (datetime.now() - last_learning_update) > timedelta(days=self.config['model_retrain_interval']):
                        print("\nRetraining models and updating market knowledge...")
                        self.learn_market_dynamics()
                        last_learning_update = datetime.now()
                    
                    if not self.can_trade():
                        print("Not allowed to trade at this time, waiting...")
                        time.sleep(10)
                        continue
                    
                    for symbol in self.config['symbols']:
                        for timeframe in self.config['timeframes']:
                            try:
                                if symbol in self.active_trades:
                                    print(f"Already have an active trade for {symbol}, skipping")
                                    continue
                                    
                                # Analyze current market
                                analysis = self.analyze_current_market(symbol, timeframe)
                                if not analysis:
                                    print(f"Could not analyze market for {symbol} on {timeframe}")
                                    continue
                                
                                print(f"\nMarket analysis for {symbol} on {timeframe}:")
                                print(f"- Current regime: {analysis['current_state']['regime']}")
                                print(f"- RSI: {analysis['current_state']['rsi']:.2f}")
                                print(f"- Volatility: {analysis['current_state']['volatility']:.6f}")
                                print(f"- Near support: {analysis['current_state']['near_support']}")
                                print(f"- Near resistance: {analysis['current_state']['near_resistance']}")
                                print(f"- Model prediction: {'Up' if analysis['prediction'] > 0.5 else 'Down'}")
                                print(f"- Model confidence: {analysis['current_state']['confidence']:.2%}")
                                print(f"- Volatility factor: {analysis['current_state']['volatility_factor']:.2f}")
                                print(f"- Directional consistency: {analysis['current_state']['directional_consistency']:.2%}")
                                
                                if analysis['similar_scenarios']:
                                    print(f"- Historical scenarios found: {analysis['similar_scenarios']['count']}")
                                    print(f"- Historical win rate: {analysis['similar_scenarios']['win_rate']:.2%}")
                                    print(f"- Historical expectancy: {analysis['similar_scenarios']['expectancy']:.6f}")
                                
                                # Check if we should trade
                                trade_conditions = [
                                    analysis['current_state']['confidence'] >= self.config['min_trade_confidence'],
                                    analysis['similar_scenarios'] is not None,
                                    analysis['similar_scenarios']['count'] >= self.config['min_historical_samples'],
                                    analysis['similar_scenarios']['win_rate'] >= self.config['min_historical_win_rate'],
                                    ((analysis['prediction'] > 0.5 and not analysis['current_state']['near_resistance']) or
                                     (analysis['prediction'] <= 0.5 and not analysis['current_state']['near_support']))
                                ]
                                
                                if all(trade_conditions):
                                    direction = 1 if analysis['prediction'] > 0.5 else -1
                                    print(f"\nAll trade conditions met for {symbol} on {timeframe}")
                                    print("Executing trade...")
                                    analysis['timeframe'] = timeframe
                                    self.execute_trade(symbol, direction, analysis)
                                else:
                                    print("\nTrade conditions not met:")
                                    if analysis['current_state']['confidence'] < self.config['min_trade_confidence']:
                                        print(f"- Confidence too low: {analysis['current_state']['confidence']:.4f} < {self.config['min_trade_confidence']}")
                                    if analysis['similar_scenarios'] is None:
                                        print("- No similar historical scenarios found")
                                    elif analysis['similar_scenarios']['count'] < self.config['min_historical_samples']:
                                        print(f"- Not enough historical samples: {analysis['similar_scenarios']['count']} < {self.config['min_historical_samples']}")
                                    elif analysis['similar_scenarios']['win_rate'] < self.config['min_historical_win_rate']:
                                        print(f"- Historical win rate too low: {analysis['similar_scenarios']['win_rate']:.2%} < {self.config['min_historical_win_rate']}")
                                    elif analysis['prediction'] > 0.5 and analysis['current_state']['near_resistance']:
                                        print("- Predicted up but near resistance")
                                    elif analysis['prediction'] <= 0.5 and analysis['current_state']['near_support']:
                                        print("- Predicted down but near support")
                            
                            except Exception as e:
                                print(f"Error processing {symbol} on {timeframe}: {str(e)}")
                                continue
                    
                    print("\nCycle completed, waiting for next iteration...")
                    time.sleep(30)
                
                except Exception as e:
                    print(f"Error in main trading loop: {str(e)}")
                    time.sleep(60)
        
        except KeyboardInterrupt:
            print("\nShutting down Market Understanding Bot...")
            self.close_all_trades()
            mt5.shutdown()


if __name__ == "__main__":
    bot = MarketUnderstandingBot()
    bot.run()