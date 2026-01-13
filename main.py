from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.models import Model
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import random
import time
import requests
import hashlib
import os

app = Flask(__name__)

# ==========================================
# ☁️ CONFIGURATION CLOUD (A REMPLIR !)
# ==========================================
# Colle l'URL de ton cluster Qdrant Cloud ici :
QDRANT_URL = "https://d7341f89-12dc-4343-a4b2-09d286409eed.europe-west3-0.gcp.cloud.qdrant.io" 

# Colle ta Clé API Secrète ici :
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.HyPB5ahmC7Z0BtS48cGx09AmO6taBVcVpewliHDtVjwy_fr0m_qdrant_cloud"

# Nom de la collection
COLLECTION_NAME = "palm_pay_prod"

# Adresse de la Blockchain (Si tu ne l'as pas mise en ligne, laisse vide ou localhost)
# Pour l'instant, si le serveur Go n'est pas en ligne, l'appli fonctionnera quand même sans blockchain.
GO_BLOCKCHAIN_URL = "http://127.0.0.1:8000/mine" 
# ==========================================

# --- 1. CONNEXION BASE DE DONNÉES ---
print("🔌 Connexion à Qdrant Cloud...")
try:
    q_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Vérification / Création de la table
    try:
        q_client.get_collection(COLLECTION_NAME)
        print("✅ Base de données trouvée !")
    except:
        print("⚠️ Création de la collection Cloud...")
        q_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1280, distance=Distance.COSINE),
        )
        print("✅ Collection créée !")
except Exception as e:
    print(f"❌ ERREUR CRITIQUE QDRANT : {e}")

# --- 2. CHARGEMENT IA ---
print("🧠 Chargement IA (MobileNetV2)...")
# On utilise le CPU pour économiser la mémoire sur Render
tf.config.set_visible_devices([], 'GPU')

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
model = Model(inputs=base_model.input, outputs=x)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Système de file d'attente pour le scanner (Mémoire vive)
current_command = {} 

# --- FONCTIONS TECHNIQUES ---
def generate_account_number():
    return f"8842-{random.randint(1000,9999)}"

def generate_embedding(palm_img):
    img = cv2.resize(palm_img, (224, 224))
    x = tf_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x, verbose=0).flatten()

def extract_palm_roi(image, landmarks):
    try:
        h, w, c = image.shape
        wrist = landmarks.landmark[0]
        middle = landmarks.landmark[9]
        angle = np.degrees(np.arctan2(wrist.y*h - middle.y*h, wrist.x*w - middle.x*w)) + 90
        M = cv2.getRotationMatrix2D((middle.x*w, middle.y*h), angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        s = int(np.sqrt((middle.x*w - wrist.x*w)**2 + (middle.y*h - wrist.y*h)**2) * 2.5)
        x1, y1 = int(middle.x*w - s//2), int(middle.y*h - s//2)
        if x1<0 or y1<0: return None
        return rotated[y1:y1+s, x1:x1+s]
    except: return None

# --- ROUTES SCANNER / TÉLÉCOMMANDE ---

@app.route('/mobile/start_enrollment', methods=['POST'])
def start_enrollment():
    global current_command
    data = request.json
    acc = data.get('account_number')
    current_command = {"status": "WAITING_FOR_SCANNER", "account": acc, "timestamp": time.time()}
    print(f"📱 ORDRE : Activer Scanner pour {acc}")
    return jsonify({"status": "SUCCESS"})

@app.route('/scanner/check_command', methods=['GET'])
def check_command():
    global current_command
    # Expire après 30 secondes
    if current_command and (time.time() - current_command.get('timestamp', 0) < 30):
        return jsonify(current_command)
    return jsonify({"status": "IDLE"})

@app.route('/scanner/complete', methods=['POST'])
def scanner_complete():
    global current_command
    current_command = {} 
    
    if 'image' not in request.files: return jsonify({"error": "No image"}), 400
    acc_num = request.form.get('account_number')
    
    file = request.files['image'].read()
    npimg = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    
    if not res.multi_hand_landmarks: return jsonify({"error": "Pas de main"}), 400
    palm = extract_palm_roi(img, res.multi_hand_landmarks[0])
    if palm is None: return jsonify({"error": "Cadrage raté"}), 400
    
    vector = generate_embedding(palm).tolist()
    
    # Tentative Blockchain (Optionnel pour le Cloud si Go n'est pas là)
    try:
        hand_token_hash = hashlib.sha256(str(vector).encode()).hexdigest()
        requests.post(GO_BLOCKCHAIN_URL, json={"wallet_id": acc_num, "hand_token": hand_token_hash}, timeout=1)
    except:
        print("⚠️ Blockchain injoignable (Normal si pas déployée)")

    # Mise à jour Cloud DB
    records, _ = q_client.scroll(collection_name=COLLECTION_NAME, limit=100, with_payload=True)
    target_id = None
    for r in records:
        if r.payload.get('account_number') == acc_num:
            target_id = r.id
            payload_update = r.payload
            payload_update["biometrics_active"] = True
            
            q_client.upsert(
                collection_name=COLLECTION_NAME,
                points=[PointStruct(id=target_id, vector=vector, payload=payload_update)]
            )
            return jsonify({"status": "SUCCESS"})
            
    return jsonify({"error": "Compte introuvable"}), 404

# --- ROUTES APPLI MOBILE ---

@app.route('/mobile/register', methods=['POST'])
def mobile_register():
    try:
        data = request.json
        name = data.get('name')
        password = data.get('password')
        acc_num = f"8842-{random.randint(1000,9999)}"
        user_id = str(uuid.uuid4())
        
        # Vecteur vide
        dummy_vector = np.random.rand(1280).tolist()
        
        q_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[PointStruct(
                id=user_id, 
                vector=dummy_vector, 
                payload={
                    "name": name, 
                    "balance": 0.0,
                    "account_number": acc_num,
                    "password": password,
                    "biometrics_active": False
                }
            )]
        )
        print(f"🆕 COMPTE CLOUD CRÉÉ : {acc_num}")
        return jsonify({"status": "SUCCESS", "account_number": acc_num})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/mobile/login', methods=['POST'])
def mobile_login():
    try:
        data = request.json
        acc_num = data.get('account_number')
        password = data.get('password')
        
        records, _ = q_client.scroll(collection_name=COLLECTION_NAME, limit=100, with_payload=True)
        for r in records:
            p = r.payload
            if p.get('account_number') == acc_num and str(p.get('password')) == str(password):
                return jsonify({
                    "status": "SUCCESS",
                    "name": p['name'],
                    "balance": p['balance']
                })
        return jsonify({"status": "FAIL", "message": "Identifiants incorrects"})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/mobile/refresh', methods=['POST'])
def mobile_refresh():
    try:
        acc_num = request.json.get('account_number')
        records, _ = q_client.scroll(collection_name=COLLECTION_NAME, limit=100, with_payload=True)
        for r in records:
            if r.payload.get('account_number') == acc_num:
                return jsonify({"balance": r.payload['balance']})
        return jsonify({"balance": 0.0})
    except: return jsonify({"balance": 0.0})

@app.route('/mobile/topup', methods=['POST'])
def mobile_topup():
    try:
        data = request.json
        acc_num = data.get('account_number')
        amount = float(data.get('amount', 0))
        
        records, _ = q_client.scroll(collection_name=COLLECTION_NAME, limit=100, with_payload=True)
        for r in records:
            if r.payload.get('account_number') == acc_num:
                new_balance = r.payload['balance'] + amount
                q_client.set_payload(
                    collection_name=COLLECTION_NAME,
                    points=[r.id],
                    payload={"balance": new_balance}
                )
                return jsonify({"status": "SUCCESS", "new_balance": new_balance})
        return jsonify({"status": "FAIL"})
    except: return jsonify({"status": "ERROR"}), 500

@app.route('/')
def home():
    return "PALM PAY API IS RUNNING ON CLOUD ☁️"

if __name__ == '__main__':
    # Render utilise le port 10000 par défaut
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)