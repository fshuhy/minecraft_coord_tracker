import cv2
import pytesseract
import re
import time
import numpy as np
import os

# --- Settings ---
# Tesseract installation path
TESSERACT_PATH = r'C:\Users\sh37\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Target device name
TARGET_CAMERA_NAME = "Logi C270 HD WebCam"

# ROI Setting (Region of Interest)
# 統合版の座標表示エリア。
ROI_TOP, ROI_BOTTOM = 10, 160   # 縦方向
ROI_LEFT, ROI_RIGHT = 10, 600   # 横方向

def find_camera_index(target_name):
    """
    Find camera index by device name.
    """
    print(f"Searching for camera: {target_name}...")
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cap.release()
            return i
    return 0

class CoordinateTracker:
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.running = True
        self.last_debug_time = 0 # デバッグ画像更新用のタイマー

    def extract_coordinates(self, text):
        """
        Extract coordinates from Bedrock Edition's "座標: X, Y, Z" format.
        """
        # 不要な記号やOCR誤認文字を整理
        # 「座標」という文字自体が誤読されても、コロン以降の数字を狙う
        clean_text = text.replace(' ', '').replace('：', ':').replace(';', ':').replace('|', '').replace('I', '1').replace('O', '0')
        
        # コロンがあればそれ以降のテキストを対象にする
        if ':' in clean_text:
            clean_text = clean_text.split(':')[-1]

        # 数値（マイナス、小数点含む）を抽出
        numbers = re.findall(r"[-+]?\d+\.?\d*", clean_text)
        
        if len(numbers) >= 3:
            try:
                # 最初に見つかった3つの数値をX, Y, Zとして採用
                return float(numbers[0]), float(numbers[1]), float(numbers[2])
            except ValueError:
                pass
            
        return None

    def preprocess_image(self, roi):
        """
        統合版の透過背景に合わせた画像処理。
        """
        if roi is None or roi.size == 0:
            return None
            
        # 1. グレースケール化
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 2. リサイズ (OCRが認識しやすいよう2.5倍に拡大)
        resized = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        
        # 3. メディアンフィルタでノイズ除去（透過背景のざらつきを抑える）
        blurred = cv2.medianBlur(resized, 3)
        
        # 4. 適応的二値化
        # C270の特性に合わせ、ブロックサイズを少し大きくして文字の連続性を保つ
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 13, 2
        )
        
        # 5. 白背景に黒文字に反転
        final_img = cv2.bitwise_not(thresh)
        
        # 【デバッグ用】5秒おきに画像を更新
        current_time = time.time()
        if current_time - self.last_debug_time > 5:
            cv2.imwrite("debug_roi.png", final_img)
            self.last_debug_time = current_time
        
        return final_img

    def start(self):
        if not os.path.exists(TESSERACT_PATH):
            print(f"Error: Tesseract not found at: {TESSERACT_PATH}")
            return

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        
        # カメラ解像度 1280x720
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print(f"Error: Could not open {TARGET_CAMERA_NAME}")
            return

        print(f"Bedrock Tracking started ({TARGET_CAMERA_NAME}).")
        print("Check 'debug_roi.png' to verify the text clarity.")
        print("Press Ctrl+C to stop.")

        try:
            while self.running:
                start_time = time.time()
                
                # バッファ破棄
                for _ in range(5):
                    cap.grab()
                ret, frame = cap.retrieve()
                
                if not ret:
                    break

                # ROI切り出し
                roi = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
                processed = self.preprocess_image(roi)

                if processed is not None:
                    # PSM 6: 単一のテキストブロック
                    config = r'--psm 6'
                    text = pytesseract.image_to_string(processed, lang='jpn+eng', config=config)
                    
                    coords = self.extract_coordinates(text)

                    timestamp = time.strftime('%H:%M:%S')
                    if coords:
                        x, y, z = coords
                        print(f"[{timestamp}] SUCCESS -> X:{x:8.1f} Y:{y:8.1f} Z:{z:8.1f}")
                    else:
                        raw = text.strip().replace('\n', ' ')
                        if raw:
                            print(f"[{timestamp}] Raw Output: [{raw[:40]}]")

                # ループ間隔維持
                elapsed = time.time() - start_time
                time.sleep(max(0.1, 0.5 - elapsed))

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            cap.release()
            print("Finished.")

if __name__ == "__main__":
    idx = find_camera_index(TARGET_CAMERA_NAME)
    tracker = CoordinateTracker(idx)
    tracker.start()