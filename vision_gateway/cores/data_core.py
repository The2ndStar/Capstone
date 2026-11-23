import cv2
import csv
import json
import os
import time
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

@dataclass
class ROIProfileItem:
    """Saved Region of Interest (ROI) in config/roi_profile.json."""
    roi_id: str                  # e.g., "ROI_1" or "QR_1"
    label: str                   # e.g., "Serial_Code"
    mode: str                    # "barcode", "qr", or "ocr"
    x_pct: float                 # Normalized 0.0 - 1.0
    y_pct: float
    w_pct: float
    h_pct: float
    preferred_bank: int = 1      # 1=Raw, 2=CLAHE, 3=Adaptive Threshold

@dataclass
class ScanRecord:
    """
    Shared data contract between Phase 1 (You) and Phase 2 (Other Devs):
    - You fill: timestamp, frame_id, code_id, code_type, sharpness, contrast, raw_qr_data, image_path
    - Dev 1 / Dev 2 fill: quality_status ('Good'/'Not Good'/'Error'), decoded_text, compare_status ('OK'/'NG')
    """
    timestamp: str
    frame_id: int
    code_id: str
    code_type: str
    sharpness: float
    contrast: float
    raw_qr_data: str = ""
    quality_status: str = "PENDING"
    decoded_text: str = ""
    compare_status: str = "PENDING"
    image_path: str = ""

class DataCore:
    def __init__(
        self,
        roi_profile_path: str = "config/roi_profile.json",
        preset_cache_path: str = "config/preset_cache.json",
        capture_dir: str = "captures",
        log_csv_path: str = "logs/scan_logsheet.csv",
    ):
        self.roi_profile_path = roi_profile_path
        self.preset_cache_path = preset_cache_path
        self.capture_dir = capture_dir
        self.log_csv_path = log_csv_path

        os.makedirs(os.path.dirname(self.roi_profile_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.preset_cache_path), exist_ok=True)
        os.makedirs(self.capture_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_csv_path), exist_ok=True)

        self._last_snap_time = 0.0
        self._init_logsheet()

    # --- 1. ROI PROFILE STORAGE (config/roi_profile.json) ---
    def save_roi_profiles(self, items: List[ROIProfileItem]):
        with open(self.roi_profile_path, "w") as f:
            json.dump([asdict(i) for i in items], f, indent=4)

    def load_roi_profiles(self) -> List[ROIProfileItem]:
        if not os.path.exists(self.roi_profile_path):
            return []
        with open(self.roi_profile_path, "r") as f:
            return [ROIProfileItem(**item) for item in json.load(f)]

    # --- 2. SNAP & KEEP IMAGE ---
    def snap_image(self, image: np.ndarray, tag: str = "SNAP", cooldown_sec: float = 0.0) -> Optional[str]:
        now = time.time()
        if cooldown_sec > 0 and (now - self._last_snap_time) < cooldown_sec:
            return None
        self._last_snap_time = now

        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        ms = int((now % 1) * 1000)
        filename = os.path.join(self.capture_dir, f"{tag}_{ts_str}_{ms:03d}.jpg")
        cv2.imwrite(filename, image)
        return filename

    # --- 3. PRESET CACHE (Room 1 Sync) ---
    def save_preset_cache(self, presets: Dict[str, str]):
        with open(self.preset_cache_path, "w") as f:
            json.dump(presets, f, indent=4)

    def load_preset_cache(self) -> Dict[str, str]:
        if not os.path.exists(self.preset_cache_path):
            return {}
        with open(self.preset_cache_path, "r") as f:
            return json.load(f)

    # --- 4. SCAN LOGSHEET (Room 2 Sync) ---
    def _init_logsheet(self):
        if not os.path.exists(self.log_csv_path):
            with open(self.log_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "frame_id", "code_id", "code_type",
                    "sharpness", "contrast", "raw_qr_data",
                    "quality_status", "decoded_text", "compare_status", "image_path"
                ])

    def keep_record(self, record: ScanRecord):
        with open(self.log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                record.timestamp, record.frame_id, record.code_id, record.code_type,
                f"{record.sharpness:.2f}", f"{record.contrast:.2f}", record.raw_qr_data,
                record.quality_status, record.decoded_text, record.compare_status, record.image_path
            ])