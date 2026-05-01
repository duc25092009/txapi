"""
API DỰ ĐOÁN TÀI XỈU - FIX LỖI FETCH HISTORY
"""

import requests, numpy as np, pandas as pd, warnings, os, time, threading, json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import pickle

warnings.filterwarnings('ignore')
app = Flask(__name__)
CORS(app)

# ==================== CẤU HÌNH ====================
API_URLS = [
    "https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=3959701241b686f12e01bfe9c3a319b8",
    "https://wtx.tele68.com/v1/tx/lite-sessions?cp=R&cl=R&pf=web&at=83991213bfd4c554dc94bcd98979bdc5",
]

# Proxy fallback (tự động thử từng cái)
PROXIES = [
    "",  # direct
    "https://api.allorigins.win/raw?url=",
    "https://corsproxy.io/?",
    "https://proxy.cors.sh/",
    "https://cors-anywhere.herokuapp.com/",
]

DATA_FILE = "taixiu_history.pkl"
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"
FETCH_INTERVAL = 60  # giây

# ==================== HÀM FETCH DỮ LIỆU (CÓ FALLBACK) ====================
def fetch_history_robust(max_retries=3):
    """Gọi API với nhiều URL và proxy khác nhau"""
    for retry in range(max_retries):
        for url in API_URLS:
            for proxy in PROXIES:
                try:
                    if proxy:
                        fetch_url = proxy + requests.utils.quote(url, safe='')
                    else:
                        fetch_url = url
                    
                    print(f"🔄 Đang thử: {fetch_url[:80]}...")
                    r = requests.get(fetch_url, timeout=15, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    
                    if r.status_code == 200:
                        data = r.json()
                        records = data.get("list", [])
                        if records and len(records) > 0:
                            print(f"✅ Thành công! Lấy được {len(records)} ván từ {url[:50]}")
                            return records
                    else:
                        print(f"⚠️ HTTP {r.status_code} - {url[:50]}")
                except Exception as e:
                    print(f"❌ Lỗi: {str(e)[:50]} - {proxy[:30] if proxy else 'direct'}")
                    continue
        
        if retry < max_retries - 1:
            print(f"🔄 Thử lại lần {retry + 2} sau 3 giây...")
            time.sleep(3)
    
    print("❌ Tất cả các cách đều thất bại!")
    return []

def fetch_and_store():
    """Lấy dữ liệu mới nhất và lưu vào file"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Đang fetch dữ liệu...")
    
    new_records = fetch_history_robust()
    if not new_records:
        print("⚠️ Không lấy được dữ liệu mới")
        return
    
    # Đọc dữ liệu cũ
    old_records = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'rb') as f:
                old_records = pickle.load(f)
        except:
            pass
    
    # Gộp và loại trùng theo id
    all_ids = {rec['id'] for rec in old_records}
    for rec in new_records:
        if rec['id'] not in all_ids:
            old_records.append(rec)
            all_ids.add(rec['id'])
    
    # Sắp xếp mới nhất lên đầu
    old_records.sort(key=lambda x: x['id'], reverse=True)
    
    # Giữ tối đa 2000 ván
    if len(old_records) > 2000:
        old_records = old_records[:2000]
    
    # Lưu lại
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(old_records, f)
    
    print(f"💾 Đã lưu {len(old_records)} ván (thêm {len(new_records)} ván mới)")

# ==================== FEATURE ENGINEERING ====================
def create_features(df):
    """Xây dựng features từ dữ liệu thô"""
    df['label'] = (df['resultTruyenThong'].str.upper().isin(['TAI', 'TÀI'])).astype(int)
    df['point'] = df['point'].astype(float)
    
    # Rolling windows
    for w in [3, 5, 7, 10, 15]:
        df[f'tai_ratio_{w}'] = df['label'].rolling(w).mean().shift(1)
        df[f'point_ma_{w}'] = df['point'].rolling(w).mean().shift(1)
    
    # Lags
    for lag in range(1, 10):
        df[f'label_lag_{lag}'] = df['label'].shift(lag)
        df[f'point_lag_{lag}'] = df['point'].shift(lag)
    
    # Streak
    streak = np.zeros(len(df))
    cur = 1
    for i in range(len(df)):
        if i > 0 and df['label'].iloc[i] == df['label'].iloc[i-1]:
            cur += 1
        else:
            cur = 1
        streak[i] = cur
    df['streak'] = streak.shift(1).fillna(1)
    
    # Pattern detection
    df['is_bet'] = (df['streak'] >= 3).astype(int)
    df['is_alternating'] = 0
    for i in range(4, len(df)):
        if (df['label'].iloc[i-1] != df['label'].iloc[i-2] and
            df['label'].iloc[i-2] != df['label'].iloc[i-3] and
            df['label'].iloc[i-3] != df['label'].iloc[i-4]):
            df.loc[df.index[i], 'is_alternating'] = 1
    
    df = df.dropna().reset_index(drop=True)
    return df

# ==================== MODEL ====================
class TaiXiuPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_cols = None
        self.is_trained = False
    
    def train(self, df):
        exclude = ['label', 'resultTruyenThong', 'id', 'timestamp', 'dices']
        self.feature_cols = [c for c in df.columns if c not in exclude]
        X = df[self.feature_cols].values.astype(np.float32)
        y = df['label'].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        split = int(len(df) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Ensemble Voting
        lr = LogisticRegression(max_iter=1000, C=0.5)
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
        
        self.model = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('xgb', xgb), ('gb', gb)],
            voting='soft',
            weights=[1, 2, 2, 2]
        )
        self.model.fit(X_train, y_train)
        
        from sklearn.metrics import accuracy_score
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"🎯 Độ chính xác trên test: {acc:.2%}")
        
        self.is_trained = True
        return acc
    
    def predict_proba(self, df):
        last_row = df.tail(1)[self.feature_cols].values.astype(np.float32)
        last_scaled = self.scaler.transform(last_row)
        return self.model.predict_proba(last_scaled)[0, 1]

# ==================== HUẤN LUYỆN ĐỊNH KỲ ====================
predictor = TaiXiuPredictor()
last_train_time = None

def auto_train():
    global predictor, last_train_time
    if not os.path.exists(DATA_FILE):
        print("⏳ Chưa có dữ liệu, đợi fetch...")
        return False
    
    with open(DATA_FILE, 'rb') as f:
        records = pickle.load(f)
    
    if len(records) < 100:
        print(f"⚠️ Chưa đủ dữ liệu ({len(records)}/100 ván)")
        return False
    
    df_raw = pd.DataFrame(records)
    df = create_features(df_raw)
    if len(df) < 50:
        return False
    
    predictor.train(df)
    last_train_time = datetime.now()
    return True

# ==================== BACKGROUND THREAD ====================
def background_worker():
    """Chạy ngầm: fetch data mỗi phút, train lại mỗi 30 phút"""
    last_train_check = 0
    while True:
        try:
            # Fetch dữ liệu mỗi phút
            fetch_and_store()
            
            # Kiểm tra train lại sau mỗi 30 phút
            now = time.time()
            if now - last_train_check > 1800:  # 30 phút
                print("🔄 Đang kiểm tra và huấn luyện lại model...")
                auto_train()
                last_train_check = now
        except Exception as e:
            print(f"Lỗi background: {e}")
        
        time.sleep(FETCH_INTERVAL)

# ==================== API ENDPOINTS ====================
@app.route('/predict', methods=['GET'])
def predict():
    if not predictor.is_trained:
        return jsonify({"success": False, "error": "Model chưa sẵn sàng, đang thu thập dữ liệu..."}), 202
    
    if not os.path.exists(DATA_FILE):
        return jsonify({"success": False, "error": "Chưa có dữ liệu"}), 400
    
    with open(DATA_FILE, 'rb') as f:
        records = pickle.load(f)
    
    df_raw = pd.DataFrame(records[:200])
    df = create_features(df_raw)
    
    proba = predictor.predict_proba(df)
    pred = "TÀI" if proba >= 0.5 else "XỈU"
    confidence = min(99, int(abs(proba - 0.5) * 200))
    
    return jsonify({
        "success": True,
        "prediction": pred,
        "confidence": confidence,
        "probability_tai": round(proba, 4),
        "total_rounds": len(records)
    })

@app.route('/history', methods=['GET'])
def history():
    if not os.path.exists(DATA_FILE):
        return jsonify({"success": False, "history": []})
    
    with open(DATA_FILE, 'rb') as f:
        records = pickle.load(f)
    
    return jsonify({
        "success": True,
        "history": records[:50],
        "total": len(records)
    })

@app.route('/health', methods=['GET'])
def health():
    total = 0
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as f:
            records = pickle.load(f)
            total = len(records)
    
    return jsonify({
        "status": "ok",
        "trained": predictor.is_trained,
        "total_records": total,
        "need": max(0, 100 - total)
    })

@app.route('/force_fetch', methods=['POST'])
def force_fetch():
    """Force fetch dữ liệu ngay lập tức"""
    fetch_and_store()
    auto_train()
    return jsonify({"success": True})

# ==================== KHỞI ĐỘNG ====================
if __name__ == "__main__":
    print("🚀 Khởi động API Tài Xỉu Predictor...")
    
    # Chạy background thread
    thread = threading.Thread(target=background_worker, daemon=True)
    thread.start()
    
    # Thử fetch lần đầu
    time.sleep(2)
    fetch_and_store()
    auto_train()
    
    app.run(host="0.0.0.0", port=5000)
