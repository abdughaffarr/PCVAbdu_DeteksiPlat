from flask import Flask, request, render_template, redirect, url_for, Response
import os
import cv2
import torch
import re
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO
import time

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Muat Model (Hanya sekali saat server dimulai) ---
# ... (Bagian ini sama persis seperti sebelumnya, tidak perlu diubah)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan device: {DEVICE}")

try:
    print("Memuat model deteksi plat nomor (YOLO)...")
    model_deteksi = YOLO("license_plate_detector.pt")
    print("Memuat model OCR (TrOCR)...")
    processor_ocr = TrOCRProcessor.from_pretrained("./trocr_finetuned_plat_nomor_final_v2/")
    model_ocr = VisionEncoderDecoderModel.from_pretrained("./trocr_finetuned_plat_nomor_final_v2/").to(DEVICE)
    print("Semua model berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

KODE_WILAYAH_PLAT = {
    'A': 'Banten', 'B': 'DKI Jakarta/Tangerang/Bekasi', 'D': 'Bandung', 'E': 'Cirebon', 'F': 'Bogor', 'G': 'Pekalongan', 'H': 'Semarang',
    'K': 'Pati', 'L': 'Surabaya', 'M': 'Madura', 'N': 'Malang', 'P': 'Banyuwangi', 'R': 'Banyumas', 'S': 'Bojonegoro',
    'T': 'Purwakarta', 'W': 'Sidoarjo/Gresik', 'Z': 'Garut/Tasikmalaya', 'AA': 'Kedu', 'AB': 'Yogyakarta', 'AD': 'Surakarta',
    'AE': 'Madiun', 'AG': 'Kediri', 'BL': 'Aceh', 'BB': 'Sumatera Utara Barat', 'BK': 'Sumatera Utara Timur', 'BA': 'Sumatera Barat',
    'BM': 'Riau', 'BH': 'Jambi', 'BD': 'Bengkulu', 'BG': 'Sumatera Selatan', 'BN': 'Bangka Belitung', 'BE': 'Lampung',
    'KB': 'Kalimantan Barat', 'KH': 'Kalimantan Tengah', 'KT': 'Kalimantan Timur', 'DA': 'Kalimantan Selatan', 'KU': 'Kalimantan Utara',
    'DB': 'Sulawesi Utara', 'DM': 'Gorontalo', 'DN': 'Sulawesi Tengah', 'DT': 'Sulawesi Tenggara', 'DD': 'Sulawesi Selatan',
    'DC': 'Sulawesi Barat', 'DE': 'Maluku', 'DG': 'Maluku Utara', 'PA': 'Papua', 'PB': 'Papua Barat'
}

def bersihkan_teks_plat(text):
    return re.sub(r'[^A-Z0-9]', '', text).upper()

def proses_gambar_dan_deteksi(image_path):
    # ... (Fungsi ini sama persis seperti sebelumnya, tidak perlu diubah)
    frame = cv2.imread(image_path)
    if frame is None:
        return None, "Gagal membaca file gambar.", []
    results = model_deteksi(frame)[0]
    detection_found = False
    all_results = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > 0.4:
            detection_found = True
            plat_terpotong = frame[int(y1):int(y2), int(x1):int(x2)]
            if plat_terpotong.size == 0: continue
            pil_plat = Image.fromarray(cv2.cvtColor(plat_terpotong, cv2.COLOR_BGR2RGB))
            pixel_values = processor_ocr(images=pil_plat, return_tensors="pt").pixel_values.to(DEVICE)
            generated_ids = model_ocr.generate(pixel_values, max_length=20)
            plat_teks = bersihkan_teks_plat(processor_ocr.batch_decode(generated_ids, skip_special_tokens=True)[0])
            kode_plat = re.match(r'^[A-Z]{1,2}', plat_teks)
            wilayah = KODE_WILAYAH_PLAT.get(kode_plat.group(0), "N/A") if kode_plat else "N/A"
            all_results.append({'text': plat_teks, 'region': wilayah})
            label = f"{plat_teks} ({wilayah})"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    if not detection_found:
        return frame, "Tidak ada plat nomor yang terdeteksi.", []
    return frame, "Deteksi berhasil.", all_results


# --- FUNGSI BARU UNTUK REAL-TIME WEBCAM STREAMING ---
def generate_frames():
    """Generator untuk streaming frame dari webcam dengan deteksi yang "menempel"."""
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_skip = 20
    frame_count = 0

    # Variabel BARU untuk "mengingat" deteksi terakhir
    last_detections = [] 

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_count += 1
            # Proses deteksi hanya pada frame yang ditentukan (misal: 1 dari 20 frame)
            if frame_count % frame_skip == 0:
                # Reset "ingatan" setiap kali deteksi baru akan dilakukan
                last_detections = [] 
                
                results = model_deteksi(frame)[0]
                for result in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = result
                    if score > 0.5:
                        plat_terpotong = frame[int(y1):int(y2), int(x1):int(x2)]
                        if plat_terpotong.size == 0: continue

                        pil_plat = Image.fromarray(cv2.cvtColor(plat_terpotong, cv2.COLOR_BGR2RGB))
                        
                        pixel_values = processor_ocr(images=pil_plat, return_tensors="pt").pixel_values.to(DEVICE)
                        generated_ids = model_ocr.generate(pixel_values, max_length=20)
                        plat_teks = bersihkan_teks_plat(processor_ocr.batch_decode(generated_ids, skip_special_tokens=True)[0])
                        
                        kode_plat = re.match(r'^[A-Z]{1,2}', plat_teks)
                        wilayah = KODE_WILAYAH_PLAT.get(kode_plat.group(0), "N/A") if kode_plat else "N/A"
                        
                        label = f"{plat_teks} ({wilayah})"
                        
                        # Simpan hasil deteksi ke dalam "ingatan"
                        last_detections.append({'box': (x1, y1, x2, y2), 'label': label})

            # BAGIAN BARU: Gambar ulang deteksi terakhir di SETIAP frame
            # Ini akan membuat kotak "menempel" dan tidak kedip-kedip
            if last_detections:
                for detection in last_detections:
                    box = detection['box']
                    label = detection['label']
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Encode frame ke format JPEG dan kirim ke browser
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    camera.release()

# --- Rute Halaman Web ---

@app.route('/')
def index():
    # Hanya menampilkan halaman HTML utama
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    # Fungsi ini khusus untuk menangani upload gambar
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        frame_hasil, status, results_data = proses_gambar_dan_deteksi(filepath)
        
        result_filename = "hasil_" + filename
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_filepath, frame_hasil)
        
        return render_template('index.html', 
                               result_image=result_filename, 
                               status=status,
                               results_data=results_data)
    return redirect(url_for('index'))

# --- RUTE BARU UNTUK VIDEO FEED ---
@app.route('/video_feed')
def video_feed():
    # Mengembalikan response streaming dari fungsi generate_frames
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)