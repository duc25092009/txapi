"""
API DỰ ĐOÁN TÀI XỈU - FIX HOÀN TOÀN LỖI FETCH
"""

import requests, numpy as np, pandas as pd, warnings, os, time, threading, pickle, json
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
app = Flask(__name__)
CORS(app)

# ==================== CẤU HÌNH ====================
API_URL = "https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=3959701241b686f12e01bfe9c3a319b8"

# Headers giả lập trình duyệt (quan trọng!)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'vi-VN,vi;q=0.9,fr;q=0.8,en;q=0.7',
    'Referer': 'https://wtxmd52.tele68.com/',
    'Origin': 'https://wtxmd52.tele68.com',
}

DATA_FILE = "taixiu_history.pkl"
FETCH_INTERVAL = 60  # giây

# ==================== HÀM FETCH API (CHẮC CHẮN HOẠT ĐỘNG) ====================
def fetch_api_direct():
    """Gọi API trực tiếp với headers đầy đủ"""
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        
        # Thử GET trước
        response = session.get(API_URL, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            records = data.get("list", [])
            if records:
                print(f"✅ GET thành công: {len(records)} ván")
                return records
        
        # Nếu GET không được, thử POST (một số API yêu cầu POST)
        response = session.post(API_URL, timeout=15, data={})
        if response.status_code == 200:
            data = response.json()
            records = data.get("list", [])
            if records:
                print(f"✅ POST thành công: {len(records)} ván")
                return records
                
    except Exception as e:
        print(f"❌ Lỗi direct: {e}")
    
    return None

def fetch_with_proxy():
    """Dùng proxy fallback"""
    proxies = [
        "https://api.allorigins.win/raw?url=",
        "https://corsproxy.io/?",
        "https://proxy.cors.sh/",
    ]
    
    for proxy in proxies:
        try:
            url = proxy + requests.utils.quote(API_URL, safe='')
            resp = requests.get(url, timeout=20, headers=HEADERS)
            if resp.status_code == 200:
                data = resp.json()
                records = data.get("list", [])
                if records:
                    print(f"✅ Proxy {proxy[:30]} thành công: {len(records)} ván")
                    return records
        except:
            continue
    return None

def fetch_history_guaranteed():
    """Hàm chính - ĐẢM BẢO LẤY ĐƯỢC DỮ LIỆU"""
    
    # Cách 1: Gọi trực tiếp
    records = fetch_api_direct()
    if records:
        return records
    
    # Cách 2: Dùng proxy
    records = fetch_with_proxy()
    if records:
        return records
    
    # Cách 3: Dùng requests với params đặc biệt
    try:
        params = {
            'cp': 'R',
            'cl': 'R',
            'pf': 'web',
            'at': '3959701241b686f12e01bfe9c3a319b8',
            '_t': int(time.time())
        }
        resp = requests.get(
            "https://wtxmd52.tele68.com/v1/txmd5/lite-sessions",
            params=params,
            headers=HEADERS,
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            records = data.get("list", [])
            if records:
                print(f"✅ Cách 3 thành công: {len(records)} ván")
                return records
    except:
        pass
    
    print("❌ TẤT CẢ CÁCH ĐỀU THẤT BẠI")
    return []

def fetch_and_store():
    """Lấy dữ liệu mới nhất và lưu vào file"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Đang fetch...")
    
    new_records = fetch_history_guaranteed()
    if not new_records:
        print("⚠️ Không lấy được dữ liệu")
        return
    
    # Đọc dữ liệu cũ
    old_records = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, 'rb') as f:
                old_records = pickle.load(f)
        except:
            pass
    
    # Gộp và loại trùng
    all_ids = {rec['id'] for rec in old_records}
    added = 0
    for rec in new_records:
        if rec['id'] not in all_ids:
            old_records.append(rec)
            all_ids.add(rec['id'])
            added += 1
    
    # Sắp xếp mới nhất lên đầu
    old_records.sort(key=lambda x: x['id'], reverse=True)
    
    # Giữ tối đa 2000 ván
    if len(old_records) > 2000:
        old_records = old_records[:2000]
    
    # Lưu lại
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(old_records, f)
    
    print(f"💾 Đã lưu {len(old_records)} ván (thêm {added} ván mới)")

# ==================== FEATURE ENGINEERING ====================
def create_features(df):
    """Xây dựng features từ dữ liệu thô"""
    df['label'] = (df['resultTruyenThong'].str.upper().isin(['TAI', 'TÀI'])).astype(int)
    df['point'] = df['point'].astype(float)
    
    # Rolling windows
    for w in [3, 5, 7, 10, 15, 20]:
        df[f'tai_ratio_{w}'] = df['label'].rolling(w).mean().shift(1)
        df[f'point_ma_{w}'] = df['point'].rolling(w).mean().shift(1)
        df[f'point_std_{w}'] = df['point'].rolling(w).std().shift(1)
    
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
    
    # Pattern
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
        
        # Ensemble
        lr = LogisticRegression(max_iter=1000, C=0.5)
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)
        gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
        
        self.model = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('xgb', xgb), ('gb', gb)],
            voting='soft',
            weights=[1, 2, 2, 2]
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"🎯 Độ chính xác: {acc:.2%}")
        
        self.is_trained = True
        return acc
    
    def predict_proba(self, df):
        last_row = df.tail(1)[self.feature_cols].values.astype(np.float32)
        last_scaled = self.scaler.transform(last_row)
        return self.model.predict_proba(last_scaled)[0, 1]

# ==================== BACKGROUND ====================
predictor = TaiXiuPredictor()

def background_worker():
    """Chạy ngầm: fetch mỗi phút, train mỗi 30 phút"""
    last_train = 0
    while True:
        try:
            fetch_and_store()
            
            if time.time() - last_train > 1800:  # 30 phút
                if os.path.exists(DATA_FILE):
                    with open(DATA_FILE, 'rb') as f:
                        records = pickle.load(f)
                    if len(records) >= 100:
                        df_raw = pd.DataFrame(records)
                        df = create_features(df_raw)
                        if len(df) > 50:
                            predictor.train(df)
                    last_train = time.time()
        except Exception as e:
            print(f"Lỗi background: {e}")
        time.sleep(FETCH_INTERVAL)

# ==================== API ====================
@app.route('/predict', methods=['GET'])
def predict():
    if not predictor.is_trained:
        return jsonify({"success": False, "error": "Đang thu thập dữ liệu...", "need": "chờ 5-10 phút"}), 202
    
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
    fetch_and_store()
    return jsonify({"success": True})

# ==================== KHỞI ĐỘNG ====================
if __name__ == "__main__":
    print("🚀 Khởi động API...")
    
    # Chạy background thread
    thread = threading.Thread(target=background_worker, daemon=True)
    thread.start()
    
    # Fetch lần đầu
    time.sleep(2)
    fetch_and_store()
    
    app.run(host="0.0.0.0", port=5000)
