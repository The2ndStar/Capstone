import cv2
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "model" / "box_detector.pt")


try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

@dataclass
class DetectedCode:
    """Unified Phase 1 output for ML-detected QR codes and 1D barcodes."""
    code_id: str                       # e.g., "QR_1" or "BARCODE_1"
    code_type: str                     # "qr" or "barcode"
    confidence: float                  # ML detection confidence (0.0 - 1.0)
    bbox: Tuple[int, int, int, int]    # (x, y, w, h) bounding box
    points: np.ndarray                 # 4 corner points
    crop_bgr: np.ndarray               # Cropped image ready for Phase 2
    banks: Dict[int, np.ndarray]       # {1: raw_gray, 2: clahe_gray, 3: thresh_gray}
    sharpness: float                   # Raw metric passed to Phase 2 dev
    contrast: float                    # Raw metric passed to Phase 2 dev
    raw_qr_data: str = ""              # Optional fast QR payload

class VisionCore:
    """
    Phase 1 Vision Engine (ML Localization + OpenCV Parameter Banks):
    - Uses a lightweight YOLO model to accurately locate ONLY real QR codes & 1D barcodes
    - Eliminates false positives from background stripes/objects
    - Crops each code & generates 3 OpenCV Parameter Banks for Phase 2
    """
    def __init__(
        self,
        source,
        model_path: str = DEFAULT_MODEL_PATH,
        conf_threshold: float = 0.55,
        width: int = 1280,
        height: int = 720,
        use_dshow: bool = False,
    ):
        self.source = source
        self.target_size = (width, height)
        self.use_dshow = use_dshow
        self.conf_threshold = conf_threshold
        self.cap = None
        self.frame_id = 0
        self.qr_detector = cv2.QRCodeDetector()

        # Load ML detector if weights exist
        self.ml_model: Optional[YOLO] = None
        if HAS_YOLO and os.path.exists(model_path):
            self.ml_model = YOLO(model_path)
            print(f"[VISION CORE] Loaded ML detector: {model_path}")
        else:
            print(f"[VISION CORE] ML weights not found at '{model_path}'. Using strict QR fallback until weights are added.")

        self._connect()

    def _connect(self):
        if self.cap is not None:
            self.cap.release()
        if self.use_dshow and isinstance(self.source, int):
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def grab_frame(self):
        """Returns (ok: bool, frame_id: int, normalized_bgr_frame: np.ndarray | None)."""
        if self.cap is None or not self.cap.isOpened():
            self._connect()
            return False, self.frame_id, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._connect()
            return False, self.frame_id, None

        if (frame.shape[1], frame.shape[0]) != self.target_size:
            frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)

        self.frame_id += 1
        return True, self.frame_id, frame

    @staticmethod
    def build_parameter_banks(roi_bgr: np.ndarray) -> Dict[int, np.ndarray]:
        """Generates the 3 OpenCV pre-processing banks for Phase 2."""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        blurred = cv2.GaussianBlur(clahe, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 4
        )
        return {1: gray, 2: clahe, 3: thresh}

    def auto_detect_all(self, frame_bgr: np.ndarray, padding: int = 10) -> List[DetectedCode]:
        """
        Uses the ML model to accurately detect QR codes and 1D barcodes without false positives.
        (Leaves Good / Not Good / Error grading to Phase 2).
        """
        h_f, w_f = frame_bgr.shape[:2]
        detections: List[DetectedCode] = []

        # --- PATH A: ML Model Detection (Activates once other dev adds models/barcode_qr_yolov8n.pt) ---
        if self.ml_model is not None:
            results = self.ml_model.predict(frame_bgr, conf=self.conf_threshold, verbose=False)[0]
            qr_count, bar_count = 0, 0

            for box in results.boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = results.names[cls_id].lower()
                code_type = "qr" if "qr" in cls_name else "barcode"

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x_min = max(0, x1 - padding)
                y_min = max(0, y1 - padding)
                x_max = min(w_f, x2 + padding)
                y_max = min(h_f, y2 + padding)
                w, h = x_max - x_min, y_max - y_min
                if w < 15 or h < 15:
                    continue

                if code_type == "qr":
                    qr_count += 1
                    cid = f"QR_{qr_count}"
                else:
                    bar_count += 1
                    cid = f"BARCODE_{bar_count}"

                crop = frame_bgr[y_min:y_max, x_min:x_max]
                banks = self.build_parameter_banks(crop)
                sharp = float(cv2.Laplacian(banks[1], cv2.CV_64F).var())
                cont = float(np.std(banks[1]))
                pts = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)

                detections.append(
                    DetectedCode(
                        code_id=cid,
                        code_type=code_type,
                        confidence=conf,
                        bbox=(x_min, y_min, w, h),
                        points=pts,
                        crop_bgr=crop,
                        banks=banks,
                        sharpness=sharp,
                        contrast=cont,
                    )
                )
            return detections

        # --- PATH B: Strict QR-Only Fallback (while waiting for Barcode+QR ML weights) ---
        found, points = self.qr_detector.detect(frame_bgr)
        if not found or points is None:
            # Also check un-mirrored frame in case Windows webcam flips horizontally
            flipped = cv2.flip(frame_bgr, 1)
            found, points = self.qr_detector.detect(flipped)
            if found and points is not None:
                points[:, :, 0] = w_f - points[:, :, 0]

        if found and points is not None:
            for idx, pts in enumerate(points):
                pts_int = pts.astype(int)
                x_min = max(0, int(np.min(pts_int[:, 0])) - padding)
                y_min = max(0, int(np.min(pts_int[:, 1])) - padding)
                x_max = min(w_f, int(np.max(pts_int[:, 0])) + padding)
                y_max = min(h_f, int(np.max(pts_int[:, 1])) + padding)
                w, h = x_max - x_min, y_max - y_min
                if w < 15 or h < 15:
                    continue

                crop = frame_bgr[y_min:y_max, x_min:x_max]
                banks = self.build_parameter_banks(crop)
                sharp = float(cv2.Laplacian(banks[1], cv2.CV_64F).var())
                cont = float(np.std(banks[1]))

                detections.append(
                    DetectedCode(
                        code_id=f"QR_{idx + 1}",
                        code_type="qr",
                        confidence=1.0,
                        bbox=(x_min, y_min, w, h),
                        points=pts_int,
                        crop_bgr=crop,
                        banks=banks,
                        sharpness=sharp,
                        contrast=cont,
                    )
                )

        return detections

    @staticmethod
    def draw_detected_boxes(frame_bgr: np.ndarray, detections: List[DetectedCode]) -> np.ndarray:
        """Draws clean bounding boxes with ML confidence score."""
        annotated = frame_bgr.copy()
        for det in detections:
            x, y, w, h = det.bbox
            color = (0, 255, 0) if det.code_type == "qr" else (255, 200, 0)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                annotated,
                f"{det.code_id} ({det.confidence:.2f})",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        return annotated

    def release(self):
        if self.cap is not None:
            self.cap.release()