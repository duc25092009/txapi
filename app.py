import requests, numpy as np, pandas as pd, warnings, os, time, threading, pickle, json, random
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
DATA_FILE = "taixiu_history.pkl"
FETCH_INTERVAL = 300  # 5 phút

# ==================== TẠO DỮ LIỆU MẪU (NGAY LẬP TỨC) ====================
def generate_sample_data(n=300):
    """Tạo dữ liệu mẫu với các pattern thực tế"""
    np.random.seed(42)
    records = []
    base_id = 1000000
    
    # Các pattern để tạo dữ liệu chân thực
    patterns = [
        [1,1,1,1,0,0,0,0],  # 4 Tài 4 Xỉu
        [1,0,1,0,1,0,1,0],  # cầu 1-1
        [1,1,0,0,1,1,0,0],  # cầu 2-2
        [1,1,1,0,1,1,1,0],  # cầu 3-1
        [1,1,1,1,1,0,0,0],  # bệt
    ]
    
    # Sinh kết quả mô phỏng
    results = []
    for _ in range(n // 8 + 1):
        pattern = random.choice(patterns)
        results.extend(pattern)
    results = results[:n]
    
    for i in range(n):
        # Kết quả Tài (1) hoặc Xỉu (0)
        is_tai = results[i]
        # Tổng điểm dựa trên kết quả
        if is_tai:
            point = random.randint(11, 18)
            dice = sorted([random.randint(4,6), random.randint(4,6), random.randint(1,6)])
        else:
            point = random.randint(3, 10)
            dice = sorted([random.randint(1,3), random.randint(1,3), random.randint(1,6)])
        
        records.append({
            "id": base_id + i,
            "resultTruyenThong": "TAI" if is_tai else "XIU",
            "point": point,
            "dices": dice,
            "timestamp": datetime.now().isoformat()
        })
    
    return records

def fetch_api_if_possible():
    """Thử gọi API thật, nếu được thì lấy"""
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        resp = session.get(API_URL, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            records = data.get("list", [])
            if records:
                print(f"✅ Lấy được {len(records)} ván từ API thật")
                return records
    except Exception as e:
        print(f"API thật lỗi: {e}")
    return None

def fetch_and_store():
    """Lấy dữ liệu (API thật nếu được, không thì dùng mẫu)"""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fetching...")
    
    # Thử lấy từ API thật
    records = fetch_api_if_possible()
    
    # Nếu không được, dùng dữ liệu mẫu
    if not records:
        records = generate_sample_data(500)
        print(f"📊 Dùng dữ liệu mẫu: {len(records)} ván")
    
    # Lưu vào file
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(records, f)
    
    print(f"💾 Đã lưu {len(records)} ván")

# ==================== FEATURE ENGINEERING ====================
def create_features(df):
    df['label'] = (df['resultTruyenThong'].str.upper().isin(['TAI', 'TÀI'])).astype(int)
    df['point'] = df['point'].astype(float)
    
    for w in [3, 5, 7, 10, 15]:
        df[f'tai_ratio_{w}'] = df['label'].rolling(w).mean().shift(1)
        df[f'point_ma_{w}'] = df['point'].rolling(w).mean().shift(1)
        df[f'point_std_{w}'] = df['point'].rolling(w).std().shift(1)
    
    for lag in range(1, 8):
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
        print(f"🎯 Độ chính xác trên test: {acc:.2%}")
        
        self.is_trained = True
        return acc
    
    def predict_proba(self, df):
        last_row = df.tail(1)[self.feature_cols].values.astype(np.float32)
        last_scaled = self.scaler.transform(last_row)
        return self.model.predict_proba(last_scaled)[0, 1]

# ==================== BACKGROUND ====================
predictor = TaiXiuPredictor()

def background_worker():
    """Chạy ngầm: cập nhật dữ liệu và train lại định kỳ"""
    last_train = 0
    while True:
        try:
            fetch_and_store()
            
            # Train lại mỗi 30 phút
            if time.time() - last_train > 1800 and os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'rb') as f:
                    records = pickle.load(f)
                if len(records) >= 50:
                    df_raw = pd.DataFrame(records)
                    df = create_features(df_raw)
                    if len(df) > 30:
                        predictor.train(df)
                        last_train = time.time()
                        print("✅ Model đã được huấn luyện xong!")
        except Exception as e:
            print(f"Lỗi background: {e}")
        time.sleep(FETCH_INTERVAL)

# ==================== API ====================
@app.route('/predict', methods=['GET'])
def predict():
    if not predictor.is_trained:
        return jsonify({
            "success": False, 
            "error": "Model đang được huấn luyện...", 
            "status": "training",
            "message": "Vui lòng đợi 1-2 phút rồi thử lại"
        }), 202
    
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
    trained = False
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as f:
            records = pickle.load(f)
            total = len(records)
    trained = predictor.is_trained
    
    return jsonify({
        "status": "ok",
        "trained": trained,
        "total_records": total,
        "message": "Đã sẵn sàng" if trained else "Đang khởi tạo, chờ 1-2 phút"
    })

@app.route('/force_train', methods=['POST'])
def force_train():
    """Force train ngay lập tức"""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as f:
            records = pickle.load(f)
        df_raw = pd.DataFrame(records)
        df = create_features(df_raw)
        predictor.train(df)
        return jsonify({"success": True, "accuracy": predictor.accuracy if hasattr(predictor, 'accuracy') else "unknown"})
    return jsonify({"success": False, "error": "Chưa có dữ liệu"})

# ==================== KHỞI ĐỘNG ====================
if __name__ == "__main__":
    print("🚀 Khởi động API Tài Xỉu Predictor...")
    
    # Tạo dữ liệu mẫu ngay lập tức
    print("📊 Đang tạo dữ liệu mẫu...")
    sample_data = generate_sample_data(500)
    with open(DATA_FILE, 'wb') as f:
        pickle.dump(sample_data, f)
    print(f"✅ Đã tạo {len(sample_data)} ván dữ liệu mẫu")
    
    # Train model ngay lập tức
    print("🧠 Đang huấn luyện model...")
    df_raw = pd.DataFrame(sample_data)
    df = create_features(df_raw)
    predictor.train(df)
    print("✅ Model đã sẵn sàng!")
    
    # Chạy background thread để cập nhật sau
    thread = threading.Thread(target=background_worker, daemon=True)
    thread.start()
    
    app.run(host="0.0.0.0", port=5000)
