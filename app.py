from flask import Flask, request, jsonify
from datetime import datetime
import pickle, os, numpy as np
from scipy.spatial.distance import cosine
import pandas as pd

from models import db, Attendance, Student, User
from utils import get_address_osm
from insightface.app import FaceAnalysis

# Dùng model nhẹ buffalo_lite
face_model = FaceAnalysis(name='buffalo_lite')
face_model.prepare(ctx_id=-1)  # ép chạy CPU

# --- Init Flask ---
app = Flask(__name__)
database_url = os.environ.get('DATABASE_URL')  # Render cung cấp
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url or 'sqlite:///app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

# --- Load embeddings ---
with open("embeddings.pkl","rb") as f:
    embeddings_dict = pickle.load(f)
THRESHOLD = 0.5

# --- Load students.csv ---
def load_students_csv():
    if Student.query.first(): return
    df = pd.read_csv("students.csv")
    df.rename(columns={"ID":"student_id","Name":"name","Class":"class_name"}, inplace=True)
    for _, row in df.iterrows():
        s = Student(student_id=row['student_id'], name=row['name'], class_name=row['class_name'])
        db.session.add(s)
    db.session.commit()
    print("Imported students.csv")

# --- Load default users ---
def load_users():
    if User.query.first(): return
    students = Student.query.all()
    for s in students:
        u = User(
            username=s.student_id,       # username = mã sinh viên
            password=s.student_id,       # password = mã sinh viên
            role="student",
            student_id=s.student_id
        )
        db.session.add(u)
    db.session.commit()
    print("Imported users")

# --- Routes ---
@app.route('/login', methods=['POST'])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    
    user = User.query.filter_by(username=username).first()
    if not user or user.password != password:
        return jsonify({"status":"failed", "message":"Sai tên đăng nhập hoặc mật khẩu"})
    
    return jsonify({
        "status":"success",
        "role": user.role,
        "student_id": user.student_id if user.role == "student" else None
    })

@app.route('/checkin', methods=['POST'])
def checkin():
    student_id = request.form.get("student_id")
    latitude = request.form.get("latitude", "0")
    longitude = request.form.get("longitude", "0")
    image_file = request.files.get("image")

    # Kiểm tra đầu vào
    if not student_id or not image_file:
        return jsonify({"status": "failed", "message": "Thiếu mã sinh viên hoặc ảnh", "address": ""})

    try:
        latitude = float(latitude)
        longitude = float(longitude)
        img_bytes = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (480, 480))  # giảm kích thước ảnh để tiết kiệm RAM
    except Exception as e:
        return jsonify({"status": "failed", "message": "Lỗi xử lý ảnh", "address": ""})

    # Nhận diện khuôn mặt
    faces = face_model.get(img)
    if not faces:
        return jsonify({"status": "failed", "message": "Không tìm thấy khuôn mặt", "address": ""})

    # Trích embedding từ khuôn mặt đầu tiên
    embedding = faces[0].embedding

    # So sánh với embeddings đã lưu
    best_score, best_id = 1.0, None
    for sid, emb_template in embeddings_dict.items():
        templates = [emb_template] if not isinstance(emb_template, list) else emb_template
        for te in templates:
            score = cosine(embedding, te)
            if score < best_score:
                best_score, best_id = score, sid

    if best_id is None or best_score > THRESHOLD:
        return jsonify({"status": "failed", "message": "Khuôn mặt không hợp lệ", "address": ""})

    # Lưu điểm danh
    now = datetime.now()
    address = get_address_osm(latitude, longitude)
    att = Attendance(
        student_id=best_id,
        date=now.date(),
        time=now.time(),
        status="present",
        latitude=latitude,
        longitude=longitude,
        address=address
    )
    db.session.add(att)
    db.session.commit()

    return jsonify({
        "status": "present",
        "message": "Điểm danh thành công",
        "student_id": best_id,
        "date": str(now.date()),
        "time": str(now.time()),
        "address": address
    })

    
@app.route('/attendance/history', methods=['GET'])
def history():
    student_id = request.args.get("student_id")
    records = Attendance.query.filter_by(student_id=student_id).order_by(Attendance.date, Attendance.time).all()
    return jsonify([{
        "student_id": r.student_id,
        "date": str(r.date),
        "time": str(r.time),
        "status": r.status,
        "latitude": r.latitude,
        "longitude": r.longitude,
        "address": r.address
    } for r in records])

@app.route('/attendance/history_teacher', methods=['GET'])
def history_teacher():
    class_name = request.args.get("class")
    date = request.args.get("date")
    q = db.session.query(Attendance, Student).join(Student, Attendance.student_id==Student.student_id)
    if class_name: q = q.filter(Student.class_name==class_name)
    if date: q = q.filter(Attendance.date==date)
    records = q.order_by(Attendance.date, Attendance.time).all()
    return jsonify([{
        "student_id": a.Attendance.student_id,
        "name": a.Student.name,
        "class": a.Student.class_name,
        "date": str(a.Attendance.date),
        "time": str(a.Attendance.time),
        "status": a.Attendance.status,
        "latitude": a.Attendance.latitude,
        "longitude": a.Attendance.longitude,
        "address": a.Attendance.address
    } for a in records])

@app.route('/attendance/export_excel', methods=['GET'])
def export_excel():
    class_name = request.args.get("class")
    date = request.args.get("date")
    q = db.session.query(Attendance, Student).join(Student, Attendance.student_id==Student.student_id)
    if class_name: q = q.filter(Student.class_name==class_name)
    if date: q = q.filter(Attendance.date==date)
    df = pd.DataFrame([{
        "student_id": a.Attendance.student_id,
        "name": a.Student.name,
        "class": a.Student.class_name,
        "date": str(a.Attendance.date),
        "time": str(a.Attendance.time),
        "status": a.Attendance.status,
        "latitude": a.Attendance.latitude,
        "longitude": a.Attendance.longitude,
        "address": a.Attendance.address
    } for a in q.all()])
    filename = f"export_{class_name}_{date}.xlsx"
    df.to_excel(filename, index=False)
    return jsonify({"status":"success","file":filename})

# --- Run ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        load_students_csv()
        load_users()
    app.run(host='0.0.0.0', port=5000)
