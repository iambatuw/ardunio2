# -*- coding: utf-8 -*-
"""
Yüz Tanıma Uygulaması (Tkinter)
- Sol Panel : Kameradan fotoğraf çek ve kişi olarak kaydet (örnek: "tufan")
- Sağ Panel : Canlı kamera ile tanıyor; tanıdığında Arduino'ya 'G' (yeşil + buzzer),
              tanımadığında 'R' (kırmızı + buzzer) gönderir.

Arduino bağlantısı: COM portunu UI'dan seçebilirsin (9600 baud).
"""

import os
import sys
import time
import pickle
import ssl
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None

APP_DIR = Path(__file__).parent
FACES_DIR = APP_DIR / "faces"
FACES_DIR.mkdir(exist_ok=True)
CACHE_FILE = APP_DIR / "embeddings.pkl"

# OpenCV ONNX yukleyicisi Windows'ta Turkce karakterli yollari okuyamiyor.
# Bu yuzden modelleri ASCII-guvenli bir konuma koyuyoruz (LOCALAPPDATA).
def _safe_models_dir():
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or "C:\\Temp"
    p = Path(base) / "face_app_models"
    p.mkdir(parents=True, exist_ok=True)
    # eger yine de ASCII degilse C:\face_app_models'a dus
    try:
        str(p).encode("ascii")
    except UnicodeEncodeError:
        p = Path("C:/face_app_models")
        p.mkdir(parents=True, exist_ok=True)
    return p

MODELS_DIR = _safe_models_dir()

# Git LFS dosyalari icin media.githubusercontent.com/media kullanilir.
# Yedek mirror URL'ler de denenir.
YUNET_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]
SFACE_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "https://huggingface.co/opencv/opencv_zoo/resolve/main/"
    "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
]
YUNET_PATH = MODELS_DIR / "yunet.onnx"
SFACE_PATH = MODELS_DIR / "sface.onnx"
# Beklenen minimum boyutlar (gercek dosyalar daha buyuk)
MIN_SIZES = {"yunet.onnx": 200_000, "sface.onnx": 30_000_000}


def _download(url, path, progress_cb=None):
    if progress_cb:
        progress_cb(f"Model indiriliyor: {path.name}")
    print(f"[model] indiriliyor: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    def _do_get(ctx):
        with urllib.request.urlopen(req, timeout=60, context=ctx) as r, \
                open(path, "wb") as f:
            while True:
                chunk = r.read(64 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    try:
        _do_get(ssl.create_default_context())
    except (ssl.SSLError, urllib.error.URLError) as e:
        # Windows'ta sertifika zinciri yoksa dogrulamayi atla
        print(f"[model] SSL hata, dogrulama atlanip yeniden deneniyor: {e}")
        _do_get(ssl._create_unverified_context())
    print(f"[model] tamam: {path} ({path.stat().st_size} bytes)")


def ensure_models(progress_cb=None):
    """Modeller yoksa indir. Bozuk/LFS-pointer dosyalari sil ve yeniden dene."""
    for urls, path in [(YUNET_URLS, YUNET_PATH), (SFACE_URLS, SFACE_PATH)]:
        min_size = MIN_SIZES[path.name]
        # mevcut dosya yeterli mi?
        if path.exists() and path.stat().st_size >= min_size:
            continue
        # bozuk varsa sil
        if path.exists():
            print(f"[model] bozuk dosya siliniyor: {path} "
                  f"({path.stat().st_size} bytes)")
            path.unlink()
        # mirror'lari sirayla dene
        last_err = None
        for url in urls:
            try:
                _download(url, path, progress_cb)
                if path.stat().st_size < min_size:
                    raise RuntimeError(
                        f"indirilen dosya cok kucuk ({path.stat().st_size} bytes)")
                break
            except Exception as e:
                last_err = e
                print(f"[model] hata, baska mirror denenecek: {e}")
                if path.exists():
                    path.unlink()
        else:
            raise RuntimeError(
                f"{path.name} indirilemedi. Son hata: {last_err}")


# ----------------- Yuz Tanima Motoru (YuNet + SFace) -----------------
# YuNet  : 5 nokta landmark ile yuz tespiti
# SFace  : 128-d embedding (cosine similarity)
# Esik   : 0.363 (>= => ayni kisi)
class FaceEngine:
    SFACE_COSINE_THRESHOLD = 0.50   # yuksek=daha siki tanima

    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(
            str(YUNET_PATH), "", (320, 320),
            score_threshold=0.7, nms_threshold=0.3, top_k=5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_PATH), "")
        # name -> list[np.ndarray (1,128)]
        self.embeddings = {}
        self.trained = False

    # ------- detection -------
    def detect(self, bgr):
        """Verilen kareyi olcekleyip yuzleri dondurur. Donus: array Nx15 ya da None."""
        h, w = bgr.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr)
        return faces

    @staticmethod
    def biggest(faces):
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda f: f[2] * f[3])

    # ------- embedding -------
    def embed(self, bgr, face_row):
        aligned = self.recognizer.alignCrop(bgr, face_row)
        feat = self.recognizer.feature(aligned)
        return feat  # shape (1, 128)

    def _augment(self, bgr):
        """Veri artirma: orijinal, ayna, parlaklik +/-"""
        out = [bgr]
        out.append(cv2.flip(bgr, 1))
        # parlaklik varyantlari
        for delta in (-25, 25):
            out.append(cv2.convertScaleAbs(bgr, alpha=1.0, beta=delta))
        return out

    # ------- training -------
    def train_from_folder(self, progress_cb=None):
        self.embeddings = {}
        per_counts = {}
        person_dirs = sorted([p for p in FACES_DIR.iterdir() if p.is_dir()])
        if not person_dirs:
            self.trained = False
            return False

        total_files = sum(1 for p in person_dirs for _ in p.glob("*.*"))
        done = 0

        for person_dir in person_dirs:
            embs = []
            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                    continue
                done += 1
                if progress_cb:
                    progress_cb(f"Egitim: {person_dir.name} "
                                f"({done}/{total_files})")
                try:
                    data = np.fromfile(str(img_path), dtype=np.uint8)
                    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                except Exception:
                    img = None
                if img is None:
                    continue
                # her varyant icin embedding
                for variant in self._augment(img):
                    faces = self.detect(variant)
                    face = self.biggest(faces)
                    if face is None:
                        continue
                    try:
                        emb = self.embed(variant, face)
                    except cv2.error:
                        continue
                    embs.append(emb)
            if embs:
                self.embeddings[person_dir.name] = embs
                per_counts[person_dir.name] = len(embs)

        print(f"[egitim] kisiler: {per_counts}")
        self.trained = len(self.embeddings) > 0
        if self.trained:
            self._save_cache()
        return self.trained

    # ------- cache -------
    def _save_cache(self):
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(self.embeddings, f)
        except Exception as e:
            print("[cache yazma hata]", e)

    def load_cache(self):
        if not CACHE_FILE.exists():
            return False
        try:
            with open(CACHE_FILE, "rb") as f:
                self.embeddings = pickle.load(f)
            self.trained = len(self.embeddings) > 0
            return self.trained
        except Exception as e:
            print("[cache okuma hata]", e)
            return False

    # ------- prediction -------
    def predict(self, bgr, face_row):
        """Donus: (name_or_None, best_score). Score yuksekse iyi eslesme."""
        if not self.trained:
            return None, 0.0
        try:
            emb = self.embed(bgr, face_row)
        except cv2.error:
            return None, 0.0

        best_name = None
        best_score = -1.0
        for name, refs in self.embeddings.items():
            for ref in refs:
                score = self.recognizer.match(
                    emb, ref, cv2.FaceRecognizerSF_FR_COSINE)
                if score > best_score:
                    best_score = score
                    best_name = name
        if best_score >= self.SFACE_COSINE_THRESHOLD:
            return best_name, best_score
        return None, best_score


# ----------------- Arduino Seri Yonetici -----------------
class ArduinoLink:
    def __init__(self):
        self.ser = None
        self.last_cmd = None
        self.last_time = 0


    @staticmethod
    def list_ports():
        if serial is None:
            return []
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self, port, baud=9600):
        self.close()
        if serial is None:
            raise RuntimeError("pyserial yüklü değil. 'pip install pyserial' yapın.")
        self.ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # Arduino reset bekle

    def close(self):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(b'O')
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    def send(self, cmd, force=False):
        # aynı komutu çok sık tekrar göndermemek için throttle
        now = time.time()
        if not force and cmd == self.last_cmd and (now - self.last_time) < 0.4:
            return
        self.last_cmd = cmd
        self.last_time = now
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(cmd.encode())
                self.ser.flush()
                print(f"[arduino] -> {cmd}")
            except Exception as e:
                print("Seri yazma hatası:", e)
        else:
            print(f"[arduino] BAGLI DEGIL, komut atildi: {cmd}")


# ----------------- Uygulama -----------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Yüz Tanıma Uygulaması")
        self.root.geometry("1100x620")
        self.root.configure(bg="#1e1e2e")

        # Modelleri indir
        try:
            ensure_models(progress_cb=lambda m: print(m))
        except Exception as e:
            messagebox.showerror(
                "Model indirme hatası",
                f"Modeller indirilemedi (internet?):\n{e}\n\n"
                f"Dosyaları manuel olarak şuraya koy:\n{MODELS_DIR}")
            raise

        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cam.isOpened():
            self.cam = cv2.VideoCapture(0)

        self.engine = FaceEngine()
        # Once cache, sonra klasor egitimi
        if not self.engine.load_cache():
            self.engine.train_from_folder()

        self.arduino = ArduinoLink()
        self.recognize_active = False
        self._red_lock_until = 0.0
        self._green_streak = 0
        self._last_green_name = None

        self._build_ui()
        self._update_loop()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI ----------
    def _build_ui(self):
        top = tk.Frame(self.root, bg="#1e1e2e")
        top.pack(fill="x", padx=10, pady=8)

        tk.Label(top, text="Arduino Port:", fg="white", bg="#1e1e2e",
                 font=("Segoe UI", 10, "bold")).pack(side="left")

        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(top, textvariable=self.port_var,
                                       values=ArduinoLink.list_ports(), width=12)
        self.port_combo.pack(side="left", padx=6)

        ttk.Button(top, text="Yenile", command=self._refresh_ports).pack(side="left")
        self.connect_btn = ttk.Button(top, text="Bağlan", command=self._toggle_connect)
        self.connect_btn.pack(side="left", padx=6)

        self.status_lbl = tk.Label(top, text="Arduino: bağlı değil",
                                   fg="#f38ba8", bg="#1e1e2e", font=("Segoe UI", 10, "bold"))
        self.status_lbl.pack(side="left", padx=12)

        # Iki panel
        panels = tk.Frame(self.root, bg="#1e1e2e")
        panels.pack(fill="both", expand=True, padx=10, pady=6)

        # --- SOL PANEL: Fotoğraf çekme ---
        left = tk.LabelFrame(panels, text=" 1) Fotoğraf Çek / Kayıt ",
                             fg="white", bg="#313244", font=("Segoe UI", 11, "bold"),
                             labelanchor="n", bd=2)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self.left_canvas = tk.Label(left, bg="black")
        self.left_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        left_btns = tk.Frame(left, bg="#313244")
        left_btns.pack(fill="x", pady=6)
        ttk.Button(left_btns, text="Fotoğraf Çek ve Kaydet",
                   command=self.capture_and_save).pack(side="left", padx=8)
        ttk.Button(left_btns, text="Modeli Yeniden Eğit",
                   command=self.retrain).pack(side="left", padx=8)
        self.left_info = tk.Label(left, text="Kayıtlı kişi: -",
                                  fg="#a6e3a1", bg="#313244", font=("Segoe UI", 10))
        self.left_info.pack(pady=4)
        self._refresh_persons_label()

        # --- SAĞ PANEL: Tanıma ---
        right = tk.LabelFrame(panels, text=" 2) Canlı Tanıma ",
                              fg="white", bg="#313244", font=("Segoe UI", 11, "bold"),
                              labelanchor="n", bd=2)
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self.right_canvas = tk.Label(right, bg="black")
        self.right_canvas.pack(fill="both", expand=True, padx=8, pady=8)

        right_btns = tk.Frame(right, bg="#313244")
        right_btns.pack(fill="x", pady=6)
        self.recog_btn = ttk.Button(right_btns, text="Tanımayı Başlat",
                                    command=self.toggle_recognize)
        self.recog_btn.pack(side="left", padx=8)
        self.result_lbl = tk.Label(right, text="Durum: bekleniyor",
                                   fg="#f9e2af", bg="#313244",
                                   font=("Segoe UI", 16, "bold"))
        self.result_lbl.pack(pady=10)

    def _refresh_ports(self):
        self.port_combo["values"] = ArduinoLink.list_ports()

    def _refresh_persons_label(self):
        persons = [p.name for p in FACES_DIR.iterdir() if p.is_dir()]
        self.left_info.config(text="Kayıtlı kişi: " + (", ".join(persons) if persons else "-"))

    def _toggle_connect(self):
        if self.arduino.ser and self.arduino.ser.is_open:
            self.arduino.close()
            self.status_lbl.config(text="Arduino: bağlı değil", fg="#f38ba8")
            self.connect_btn.config(text="Bağlan")
            return
        port = self.port_var.get().strip()
        if not port:
            messagebox.showwarning("Uyarı", "Bir COM port seçin.")
            return
        try:
            self.arduino.connect(port)
            self.arduino.send('O')
            self.status_lbl.config(text=f"Arduino: {port} bağlı", fg="#a6e3a1")
            self.connect_btn.config(text="Bağlantıyı Kes")
        except Exception as e:
            messagebox.showerror("Hata", f"Bağlantı hatası: {e}")

    # ---------- Kamera dongusu ----------
    def _update_loop(self):
        ok, frame = self.cam.read()
        if ok:
            frame = cv2.flip(frame, 1)
            faces = self.engine.detect(frame)

            left_frame = frame.copy()
            right_frame = frame.copy()

            # SOL: tum yuzleri cerceveler
            if faces is not None:
                for f in faces:
                    x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                    cv2.rectangle(left_frame, (x, y), (x+w, y+h), (0, 200, 255), 2)

            # SAG: tanima (en buyuk yuz)
            face = FaceEngine.biggest(faces)
            if self.recognize_active and self.engine.trained and face is not None:
                name, score = self.engine.predict(frame, face)
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                now = time.time()

                if name is not None:
                    color = (0, 220, 0)
                    label = f"{name} ({score:.2f})"
                else:
                    color = (0, 0, 220)
                    label = f"Bilinmiyor ({score:.2f})"

                cv2.rectangle(right_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(right_frame, label, (x, max(20, y-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if name is None:
                    self._green_streak = 0
                    self._last_green_name = None
                    self._red_lock_until = now + 2.0
                    self.result_lbl.config(text="Bilinmeyen kişi!", fg="#f38ba8")
                    self.arduino.send('R')
                elif now < self._red_lock_until:
                    self._green_streak = 0
                    self.result_lbl.config(text="Bilinmeyen kişi!", fg="#f38ba8")
                    self.arduino.send('R')
                else:
                    if name == self._last_green_name:
                        self._green_streak += 1
                    else:
                        self._last_green_name = name
                        self._green_streak = 1

                    if self._green_streak >= 3:
                        self.result_lbl.config(
                            text=f"Bu {name.upper()}!", fg="#a6e3a1")
                        self.arduino.send('G')
                    else:
                        self.result_lbl.config(text="Kontrol ediliyor...", fg="#f9e2af")
            elif self.recognize_active:
                # Yuz anlik kayboldu - LED'i degistirme, son durumu koru
                self.result_lbl.config(text="Yüz aranıyor...", fg="#f9e2af")
                if time.time() < self._red_lock_until:
                    self.arduino.send('R')

            self._show(self.left_canvas, left_frame)
            self._show(self.right_canvas, right_frame)

        self.root.after(30, self._update_loop)

    def _show(self, widget, frame_bgr):
        h = widget.winfo_height() or 360
        w = widget.winfo_width() or 480
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img.thumbnail((w, h))
        tk_img = ImageTk.PhotoImage(img)
        widget.configure(image=tk_img)
        widget.image = tk_img

    # ---------- Eylemler ----------
    def capture_and_save(self):
        name = simpledialog.askstring(
            "Kişi Adı",
            "Bu kişinin adı (örnek: tufan):\n\n"
            "Onayladıktan sonra ~4 saniye boyunca 50 fotoğraf otomatik\n"
            "çekilecek. Lütfen başınızı yavaşça sağa, sola, yukarı, aşağı\n"
            "çevirin; farklı mimikler yapın.",
            parent=self.root)
        if not name:
            return
        name = name.strip().lower().replace(" ", "_")
        person_dir = FACES_DIR / name
        person_dir.mkdir(exist_ok=True)
        start_idx = len(list(person_dir.glob("*.jpg")))

        for i in (3, 2, 1):
            self.result_lbl.config(text=f"{i}...", fg="#f9e2af")
            self.root.update()
            time.sleep(0.7)

        target_count = 50
        saved = 0
        attempts = 0
        max_attempts = 400
        while saved < target_count and attempts < max_attempts:
            attempts += 1
            ok, frame = self.cam.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            faces = self.engine.detect(frame)
            face = FaceEngine.biggest(faces)
            if face is None:
                self.result_lbl.config(
                    text=f"Yüz aranıyor... ({saved}/{target_count})",
                    fg="#f38ba8")
                self.root.update()
                time.sleep(0.03)
                continue

            out_path = person_dir / f"{start_idx + saved:03d}.jpg"
            ok_enc, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok_enc:
                continue
            try:
                buf.tofile(str(out_path))
            except Exception as e:
                print("[kayit hata]", e)
                continue
            if not out_path.exists() or out_path.stat().st_size == 0:
                continue
            saved += 1

            x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{saved}/{target_count}", (x, max(20, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            self._show(self.left_canvas, frame)
            self.result_lbl.config(text=f"Çekiliyor... {saved}/{target_count}",
                                   fg="#a6e3a1")
            self.root.update()
            time.sleep(0.07)

        self.result_lbl.config(text="Durum: bekleniyor", fg="#f9e2af")
        if saved == 0:
            messagebox.showerror("Hata", "Hiç yüz yakalanamadı.")
            return
        messagebox.showinfo(
            "Kayıt",
            f"'{name}' için {saved} fotoğraf kaydedildi.\n"
            f"Toplam: {start_idx + saved}\n\n"
            f"Model şimdi eğitilecek (birkaç saniye sürebilir).")
        self._refresh_persons_label()
        self.retrain(silent=True)

    def retrain(self, silent=False):
        # Modal "lutfen bekleyin" penceresi
        prog = tk.Toplevel(self.root)
        prog.title("Eğitim")
        prog.geometry("420x90")
        prog.transient(self.root)
        prog.configure(bg="#1e1e2e")
        prog_lbl = tk.Label(prog, text="Model eğitiliyor...",
                            fg="#cdd6f4", bg="#1e1e2e",
                            font=("Segoe UI", 10))
        prog_lbl.pack(pady=20, padx=10)
        prog.update()

        def upd(msg):
            prog_lbl.config(text=msg)
            prog.update()

        try:
            ok = self.engine.train_from_folder(progress_cb=upd)
        except Exception as e:
            prog.destroy()
            messagebox.showerror("Eğitim Hatası", f"Model eğitilemedi:\n{e}")
            return
        prog.destroy()

        if not silent:
            if ok:
                kisiler_str = ", ".join(
                    f"{n} ({len(v)} embedding)"
                    for n, v in self.engine.embeddings.items())
                messagebox.showinfo(
                    "Eğitim",
                    f"Model başarıyla eğitildi.\nKişiler: {kisiler_str}"
                )
            else:
                messagebox.showwarning(
                    "Eğitim",
                    "Eğitim için hiç kayıtlı yüz bulunamadı.\n"
                    "Önce 'Fotoğraf Çek ve Kaydet' ile en az 1 fotoğraf alın."
                )

    def toggle_recognize(self):
        if not self.engine.trained:
            messagebox.showwarning("Uyarı", "Önce en az 1 kişi kaydedip modeli eğitin.")
            return
        self.recognize_active = not self.recognize_active
        self._red_lock_until = 0.0
        self._green_streak = 0
        self._last_green_name = None
        self.recog_btn.config(text="Tanımayı Durdur" if self.recognize_active
                              else "Tanımayı Başlat")
        if not self.recognize_active:
            self.arduino.send('O')
            self.result_lbl.config(text="Durum: bekleniyor", fg="#f9e2af")
        else:
            self.arduino.send('O')

    def on_close(self):
        try:
            self.arduino.send('O')
        except Exception:
            pass
        self.arduino.close()
        if self.cam:
            self.cam.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        style.theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
