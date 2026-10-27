from pathlib import Path
import sqlite3, uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
UPLOADS = STATIC / "uploads"
PROCESSED = STATIC / "processed"
MODEL_PATH = BASE / "models" / "eye_disease.keras"
DB_PATH = BASE / "user_data.db"
ALLOWED = {"png", "jpg", "jpeg", "bmp"}
CLASS_NAMES = ["Cataract", "Glaucoma", "Mild diabetic retinopathy", "Moderate diabetic retinopathy", "Normal", "Proliferative diabetic retinopathy", "Diabetic eye disease", "Severe diabetic retinopathy"]
REMEDIES = {
 "Cataract": ["Arrange an eye examination with an ophthalmologist.", "Treatment depends on a professional clinical assessment."],
 "Glaucoma": ["Arrange a prompt eye examination; glaucoma needs professional diagnosis and follow-up.", "Treatment may include prescribed eye drops, laser treatment, or surgery."],
 "Mild diabetic retinopathy": ["Discuss the result with an eye-care professional.", "Maintain diabetes care and attend regular retinal examinations."],
 "Moderate diabetic retinopathy": ["Arrange follow-up with an eye-care professional."],
 "Normal": ["Continue routine eye examinations, especially if you have diabetes or vision changes."],
 "Proliferative diabetic retinopathy": ["Seek prompt assessment by a retinal specialist."],
 "Diabetic eye disease": ["Arrange a comprehensive eye examination."],
 "Severe diabetic retinopathy": ["Seek prompt assessment by an ophthalmologist or retinal specialist."]
}
app = Flask(__name__)
app.secret_key = "change-this-key-before-deployment"
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
UPLOADS.mkdir(parents=True, exist_ok=True); PROCESSED.mkdir(parents=True, exist_ok=True)

def db(): return sqlite3.connect(DB_PATH)
def init_db():
    with db() as con:
        con.execute("CREATE TABLE IF NOT EXISTS user (name TEXT PRIMARY KEY, password_hash TEXT NOT NULL, mobile TEXT, email TEXT)")
def allowed(name): return "." in name and name.rsplit(".",1)[1].lower() in ALLOWED
def get_model():
    if not MODEL_PATH.exists(): return None
    try:
        from tensorflow.keras.models import load_model
        return load_model(MODEL_PATH)
    except Exception:
        app.logger.exception("Model could not be loaded")
        return None

@app.route("/")
def index(): return render_template("index.html", msg=request.args.get("msg"))
@app.route("/userlog", methods=["GET","POST"])
def userlog():
    if request.method == "GET":
        if session.get("user"): return render_template("userlog.html", username=session["user"])
        return redirect(url_for("index"))
    name=request.form.get("name", "").strip(); password=request.form.get("password", "")
    with db() as con: row=con.execute("SELECT name,password_hash FROM user WHERE name=?", (name,)).fetchone()
    if not row or not check_password_hash(row[1], password): return render_template("index.html", msg="Sorry, incorrect credentials provided. Please try again.")
    session["user"]=row[0]
    return redirect(url_for("userlog"))
@app.route("/userreg", methods=["POST"])
def userreg():
    name=request.form.get("name", "").strip(); password=request.form.get("password", "")
    email=request.form.get("email", "").strip(); mobile=request.form.get("phone", "").strip()
    if not name or len(password)<6: return render_template("index.html", msg="Enter a username and a password of at least 6 characters.")
    try:
        with db() as con: con.execute("INSERT INTO user(name,password_hash,mobile,email) VALUES(?,?,?,?)", (name,generate_password_hash(password),mobile,email))
    except sqlite3.IntegrityError: return render_template("index.html", msg="That username is already registered.")
    return render_template("index.html", msg="Successfully registered. Please sign in.")
@app.route("/image", methods=["POST"])
def image():
    if not session.get("user"): return redirect(url_for("index", msg="Please sign in first."))
    file=request.files.get("filename")
    if not file or not file.filename: return render_template("userlog.html", msg="Please choose an image.")
    if not allowed(file.filename): return render_template("userlog.html", msg="Use a PNG, JPG, JPEG, or BMP image.")
    token=uuid.uuid4().hex; safe=secure_filename(file.filename)
    saved=UPLOADS/f"{token}_{safe}"; file.save(saved)
    image=cv2.imread(str(saved), cv2.IMREAD_COLOR)
    if image is None: return render_template("userlog.html", msg="The selected file is not a readable image.")
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); edges=cv2.Canny(image,100,200)
    _, threshold=cv2.threshold(gray,128,255,cv2.THRESH_BINARY)
    kernel=np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    sharpened=cv2.filter2D(image,-1,kernel)
    for name, arr in [("gray",gray),("edges",edges),("threshold",threshold),("sharpened",sharpened)]: cv2.imwrite(str(PROCESSED/f"{token}_{name}.jpg"),arr)
    urls={"ImageDisplay":url_for("static",filename=f"uploads/{saved.name}"),
          "ImageDisplay1":url_for("static",filename=f"processed/{token}_gray.jpg"),
          "ImageDisplay2":url_for("static",filename=f"processed/{token}_edges.jpg"),
          "ImageDisplay3":url_for("static",filename=f"processed/{token}_threshold.jpg"),
          "ImageDisplay4":url_for("static",filename=f"processed/{token}_sharpened.jpg")}
    model=get_model()
    if model is None:
        return render_template("userlog.html", msg="Image processing completed, but the trained model is missing. Run python train_model.py, then try again.", **urls)
    resized=cv2.resize(image,(50,50)).astype("float32")/255.0
    probs=model.predict(np.expand_dims(resized,axis=0),verbose=0)[0]
    label=CLASS_NAMES[int(np.argmax(probs))]
    accuracy=f"Model score: {float(np.max(probs))*100:.2f}% (not a clinically validated accuracy)"
    return render_template("userlog.html", status=label.lower(), accuracy=accuracy, remedie=f"Suggested next steps for {label}:", remedie1=REMEDIES.get(label,[]), **urls)
@app.route("/logout")
def logout(): session.clear(); return redirect(url_for("index"))
@app.errorhandler(413)
def too_large(_): return render_template("userlog.html", msg="Image exceeds the 10 MB upload limit.") , 413
init_db()
if __name__ == "__main__": app.run(debug=True)
