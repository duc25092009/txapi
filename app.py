"""
TÀI XỈU PREDICTOR API - TỰ ĐỘNG THU THẬP DỮ LIỆU MỖI 1 PHÚT
"""

import requests, numpy as np, pandas as pd, warnings, os, pickle, threading, time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
app = Flask(__name__)
CORS(app)

# ==================== CẤU HÌNH ====================
API_URL = "https://wtxmd52.tele68.com/v1/txmd5/lite-sessions?cp=R&cl=R&pf=web&at=3959701241b686f12e01bfe9c3a319b8"
DATA_FILE = "taixiu_history.pkl"
MODEL_FILE = "model.pkl"
SCALER_FILE = "scaler.pkl"

# ==================== LẤY VÀ LƯU DỮ LIỆU ====================
def fetch_and_store():
    """Gọi API, lưu lịch sử vào file (gộp với dữ liệu cũ)"""
    try:
        r = requests.get(API_URL, timeout=10)
        data = r.json()
        new_records = data.get("list", [])
        if not new_records:
            return
        
        # Load dữ liệu cũ
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'rb') as f:
                old_records = pickle.load(f)
            # Tạo dict id để tránh trùng
            old_ids = {rec['id'] for rec in old_records}
            unique_new = [rec for rec in new_records if rec['id'] not in old_ids]
            if unique_new:
                all_records = unique_new + old_records
                all_records.sort(key=lambda x: x['id'], reverse=True)
                with open(DATA_FILE, 'wb') as f:
                    pickle.dump(all_records[:5000], f)
                print(f"   ✅ Đã thêm {len(unique_new)} ván mới. Tổng: {len(all_records[:5000])}")
        else:
            with open(DATA_FILE, 'wb') as f:
                pickle.dump(new_records[:5000], f)
            print(f"   ✅ Đã tạo file mới với {len(new_records)} ván")
    except Exception as e:
        print(f"   ❌ Lỗi fetch: {e}")

# ==================== FEATURE ENGINEERING ====================
def create_features(df):
    """Xây dựng features đơn giản nhưng hiệu quả"""
    df['label'] = (df['resultTruyenThong'].str.upper().isin(['TAI', 'TÀI'])).astype(int)
    df['point'] = df['point'].astype(float)
    
    # Tỷ lệ Tài và trung bình điểm trong cửa sổ
    for w in [3, 5, 7, 10, 15, 20, 30]:
        df[f'tai_ratio_{w}'] = df['label'].rolling(w).mean().shift(1)
        df[f'point_ma_{w}'] = df['point'].rolling(w).mean().shift(1)
        df[f'point_std_{w}'] = df['point'].rolling(w).std().shift(1)
    
    # Lag labels
    for lag in range(1, 11):
        df[f'label_lag_{lag}'] = df['label'].shift(lag)
        df[f'point_lag_{lag}'] = df['point'].shift(lag)
    
    # Streak
    streak = []
    cur = 1
    for i in range(len(df)):
        if i > 0 and df['label'].iloc[i] == df['label'].iloc[i-1]:
            cur += 1
        else:
            cur = 1
        streak.append(cur)
    df['streak'] = np.array(streak).shift(1).fillna(1)
    
    # Pattern đan xen
    df['alt'] = 0
    for i in range(4, len(df)):
        if (df['label'].iloc[i-1] != df['label'].iloc[i-2] and
            df['label'].iloc[i-2] != df['label'].iloc[i-3] and
            df['label'].iloc[i-3] != df['label'].iloc[i-4]):
            df.loc[df.index[i], 'alt'] = 1
    
    df = df.dropna().reset_index(drop=True)
    return df

# ==================== HUẤN LUYỆN MODEL ====================
class TaiXiuPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False
    
    def train(self, df):
        exclude = ['label', 'resultTruyenThong', 'id', 'timestamp', 'dices']
        self.feature_cols = [c for c in df.columns if c not in exclude]
        X = df[self.feature_cols].values.astype(np.float32)
        y = df['label'].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Chia thời gian
        split = int(len(df) * 0.8)
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        self.model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)
        
        acc = accuracy_score(y_test, self.model.predict(X_test))
        print(f"🎯 Độ chính xác model: {acc:.2%}")
        self.is_trained = True
        return acc
    
    def predict_proba(self, df):
        X_new = df.tail(1)[self.feature_cols].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_new)
        proba = self.model.predict_proba(X_scaled)[0, 1]
        
        # Điều chỉnh bằng streak
        streak = df['streak'].values[-1]
        adjustment = 0.0
        if streak >= 5:
            adjustment = -0.2 if df['label'].iloc[-1] == 1 else 0.2
        elif streak >= 3:
            adjustment = -0.1 if df['label'].iloc[-1] == 1 else 0.1
        elif df['alt'].iloc[-1] == 1:
            adjustment = 0.15 if df['label'].iloc[-1] == 0 else -0.15
        
        final = np.clip(proba + adjustment, 0.05, 0.95)
        return final, streak

# ==================== BACKGROUND FETCH THREAD ====================
def background_fetcher():
    """Mỗi 60 giây chạy một lần để cập nhật dữ liệu"""
    while True:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🔄 Đang cập nhật lịch sử...")
            fetch_and_store()
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
        time.sleep(60)

# ==================== KHỞI TẠO MODEL ====================
predictor = TaiXiuPredictor()

def init_model():
    """Load hoặc train model từ dữ liệu đã lưu"""
    global predictor
    if not os.path.exists(DATA_FILE):
        print("⚠️ Chưa có dữ liệu, sẽ fetch lần đầu...")
        fetch_and_store()
    
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'rb') as f:
            records = pickle.load(f)
        if len(records) > 100:
            df_raw = pd.DataFrame(records)
            df = create_features(df_raw)
            predictor.train(df)
        else:
            print("⚠️ Dữ liệu chưa đủ (cần >100 ván)")
    else:
        print("❌ Không thể khởi tạo model do thiếu dữ liệu")

# ==================== API ENDPOINTS ====================
@app.route('/predict', methods=['GET'])
def predict():
    if not predictor.is_trained:
        return jsonify({"success": False, "error": "Model chưa sẵn sàng"}), 503
    
    # Lấy dữ liệu mới nhất từ file (đã được background thread cập nhật)
    with open(DATA_FILE, 'rb') as f:
        records = pickle.load(f)
    df_raw = pd.DataFrame(records[:500])
    if len(df_raw) < 50:
        return jsonify({"success": False, "error": "Không đủ dữ liệu"}), 400
    
    df = create_features(df_raw)
    proba, streak = predictor.predict_proba(df)
    pred = "TÀI" if proba >= 0.5 else "XỈU"
    confidence = min(99, int(abs(proba - 0.5) * 200))
    
    return jsonify({
        "success": True,
        "prediction": pred,
        "confidence": confidence,
        "probability_tai": round(proba, 4),
        "streak": int(streak),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/history', methods=['GET'])
def history():
    if not os.path.exists(DATA_FILE):
        return jsonify({"success": False, "history": []})
    with open(DATA_FILE, 'rb') as f:
        records = pickle.load(f)
    return jsonify({"success": True, "history": records[:50]})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "trained": predictor.is_trained,
        "total_records": len(pickle.load(open(DATA_FILE, 'rb'))) if os.path.exists(DATA_FILE) else 0
    })

# ==================== START ====================
if __name__ == "__main__":
    print("🚀 Khởi động API với background fetcher (1 phút/lần)...")
    # Chạy background thread
    fetcher_thread = threading.Thread(target=background_fetcher, daemon=True)
    fetcher_thread.start()
    # Khởi tạo model
    init_model()
    # Chạy Flask
    app.run(host="0.0.0.0", port=5000)
