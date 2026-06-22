import os
import sys
import uuid
import subprocess
from flask import (
    Flask, request, render_template, send_from_directory,
    redirect, url_for, flash, jsonify
)
from werkzeug.utils import secure_filename

from predict_tb import run_pipeline, OUTPUT_DIR, BASE_DIR

# ============================================================
# FLASK APP SETUP
# ============================================================

app = Flask(__name__)
app.secret_key = "tb-screening-dev-key"  # only used for flash messages

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ============================================================
# ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # ---- Validate patient fields ----
    patient_name = request.form.get("patient_name", "").strip()
    patient_id = request.form.get("patient_id", "").strip()
    gender = request.form.get("gender", "").strip()
    age = request.form.get("age", "").strip()
    referred_by = request.form.get("referred_by", "").strip()
    notes = request.form.get("notes", "").strip()

    errors = []
    if not patient_name:
        errors.append("Patient name is required.")
    if not patient_id:
        errors.append("Patient ID is required.")
    if not gender:
        errors.append("Gender is required.")
    if not age:
        errors.append("Age is required.")
    else:
        try:
            age_int = int(age)
            if age_int <= 0 or age_int > 130:
                errors.append("Age must be a realistic number.")
        except ValueError:
            errors.append("Age must be a number.")

    # ---- Validate file ----
    if "xray" not in request.files:
        errors.append("Please upload a chest X-ray image.")
    else:
        file = request.files["xray"]
        if file.filename == "":
            errors.append("Please upload a chest X-ray image.")
        elif not allowed_file(file.filename):
            errors.append("X-ray must be a PNG, JPG, JPEG, or BMP image.")

    if errors:
        for e in errors:
            flash(e, "error")
        return redirect(url_for("index"))

    # ---- Save uploaded file ----
    file = request.files["xray"]
    ext = file.filename.rsplit(".", 1)[1].lower()
    safe_id = secure_filename(patient_id) or uuid.uuid4().hex[:8]
    saved_name = f"{safe_id}_{uuid.uuid4().hex[:8]}.{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(saved_path)

    # ---- Run pipeline ----
    try:
        result = run_pipeline(
            image_path=saved_path,
            patient_name=patient_name,
            patient_id=patient_id,
            gender=gender,
            age=age,
            referred_by=referred_by,
            notes=notes,
        )
    except Exception as e:
        flash(f"Error while processing X-ray: {e}", "error")
        return redirect(url_for("index"))

    # ---- Render results ----
    heatmap_filename = os.path.basename(result["heatmap_path"])
    report_filename = os.path.basename(result["report_path"])

    return render_template(
        "result.html",
        patient=result["patient"],
        pred_label=result["pred_label"],
        pred_confidence=result["pred_confidence"],
        confidence_tb=result["confidence_tb"],
        confidence_healthy=result["confidence_healthy"],
        risk_level=result["risk_level"],
        lung_coverage_pct=result["lung_coverage_pct"],
        heatmap_filename=heatmap_filename,
        report_filename=report_filename,
    )


@app.route("/open-3d-viewer", methods=["POST"])
def open_3d_viewer():
    """
    Launches tb_portal_viewer.py (PyVista, real STL anatomy) as a
    separate desktop process. This only works when the Flask server
    and the browser are running on the SAME machine (e.g. local
    development on your own laptop) — it cannot open a desktop window
    on a different machine than the one running this server.
    """
    viewer_script = os.path.join(BASE_DIR, "tb_portal_viewer.py")

    if not os.path.exists(viewer_script):
        return jsonify({
            "ok": False,
            "error": f"tb_portal_viewer.py not found at {viewer_script}"
        }), 404

    try:
        # Popen (not run) so this returns immediately and doesn't block
        # the Flask request while the PyVista window stays open.
        subprocess.Popen(
            [sys.executable, viewer_script],
            cwd=BASE_DIR,
        )
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/outputs/<path:filename>")
def serve_output(filename):
    """Serve generated files (heatmap image, PDF report, 3D viewer HTML)."""
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5050)