import cv2
import time
from cores.vision_core import VisionCore
from cores.data_core import DataCore, ScanRecord

def main():
    vc = VisionCore(source=0, use_dshow=True)
    dc = DataCore()

    print("=== Auto QR & Barcode Detection Check ===")
    print("Hold any QR code or 1D Barcode in front of your PC webcam.")
    print("Controls: [s] Snap & Keep | [q] Quit")

    while True:
        ok, frame_id, frame = vc.grab_frame()
        if not ok:
            time.sleep(0.2)
            continue

        # 1. VisionCore automatically detects BOTH QR codes and 1D Barcodes
        detections = vc.auto_detect_all(frame)

        # 2. Draw automatic boxes (Green = QR, Cyan = Barcode)
        display = vc.draw_detected_boxes(frame, detections)

        # Show cropped Parameter Banks for the first detected code
        if detections:
            first = detections[0]
            bank_strip = cv2.hconcat([first.banks[1], first.banks[2], first.banks[3]])
            cv2.imshow("Auto-Cropped Banks (1:Raw | 2:CLAHE | 3:Thresh)", bank_strip)

        cv2.putText(display, f"Detected Codes: {len(detections)}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.imshow("Phase 1 - Auto QR & Barcode Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and detections:
            for det in detections:
                img_path = dc.snap_image(det.crop_bgr, tag=det.code_id)
                dc.keep_record(ScanRecord(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    frame_id=frame_id,
                    code_id=det.code_id,
                    code_type=det.code_type,
                    sharpness=det.sharpness,
                    contrast=det.contrast,
                    raw_qr_data=det.raw_qr_data,
                    image_path=img_path or ""
                ))
            print(f"[KEPT] Snapped & logged {len(detections)} code crop(s) for Phase 2.")

        elif key == ord('q'):
            break

    vc.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()