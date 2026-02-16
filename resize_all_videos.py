import cv2
import os

# Kaynak klasör - şu anki dizin
input_dir = r"X:\410SERV\AG0 McMahon\Özge\cutting_legs\C-"
# Çıktı klasörü
output_dir = r"X:\410SERV\AG0 McMahon\Özge\cutting_legs_resized\C-"

TARGET_W = 3240
TARGET_H = 2890

# Tüm mp4 dosyalarını bul (alt klasörler dahil)
video_files = []
for root, dirs, files in os.walk(input_dir):
    for f in files:
        if f.lower().endswith(".mp4"):
            video_files.append(os.path.join(root, f))

print(f"Toplam {len(video_files)} video bulundu\n")

for i, filepath in enumerate(video_files):
    rel_path = os.path.relpath(filepath, input_dir)
    out_path = os.path.join(output_dir, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if os.path.exists(out_path):
        print(f"[{i+1}/{len(video_files)}] ZATEN VAR, ATLANIYOR: {rel_path}")
        continue

    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print(f"[{i+1}/{len(video_files)}] ACILAMADI: {rel_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[{i+1}/{len(video_files)}] {rel_path}")
    print(f"  {orig_w}x{orig_h} -> {TARGET_W}x{TARGET_H}, {total} frame, {fps:.1f} fps")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (TARGET_W, TARGET_H))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        writer.write(resized)
        count += 1
        if count % 500 == 0:
            print(f"  {count}/{total} frame")

    cap.release()
    writer.release()
    print(f"  Tamamlandi! ({count} frame)\n")

print(f"\nBitti! Tüm videolar: {output_dir}")