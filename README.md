# Raspberry Pi Universal Industrial Vision Gateway

An autonomous, Ethernet-based **Smart Vision Gateway Module** built on a Raspberry Pi. Modeled after commercial smart vision readers (such as the Keyence SR-X series), this system acts as a universal edge processing hub that connects to **any external camera capable of transmitting video over Ethernet**—including IP CCTV cameras, PoE cameras, Ethernet-bridged mobile devices, or networked camera modules—via standard IP streaming protocols (`RTSP`, `HTTP/MJPEG`, or `WebRTC`)[cite: 1].

Instead of relying on proprietary camera sensors or an external PLC, all image processing, multi-code barcode/OCR decoding, master preset comparison, read-quality grading, and data synchronization terminate directly on the Raspberry Pi[cite: 1].

---

## Core Architecture & Key Features

* **Universal Ethernet Camera Ingestion:** Unified URL-based video receiver that ingests and normalizes live streams (varying resolutions and frame rates) from any Ethernet-capable IP camera into a standard `1280x720` processing pipeline[cite: 1].
* **Auto Box Localization & No-Code "Draw-a-Box" Visual Training (Barcode & OCR):**
  * Automatically locates 1D barcodes and 2D QR codes using a lightweight ML detector (`box_detector.pt`), or lets operators freeze a frame on a local Web UI (`localhost` / static Ethernet IP) to drag custom Region of Interest (ROI) boxes and run 1-click Auto-Tune[cite: 1].
* **Software Parameter Banks & Alternate Cycling:** Software pre-processing profiles (**Bank 1:** Raw Grayscale, **Bank 2:** CLAHE Contrast Enhancement, **Bank 3:** Gaussian Denoise + Adaptive Thresholding) cycle automatically to read low-contrast, reflective, or damaged codes[cite: 1].
* **Edge-Only Processing & Cloud "Two-Room" Sync (No PLC):**
  * **Pulls Presets (Room 1):** Fetches registered master preset values over Ethernet and caches them locally (`config/preset_cache.json`) for low-latency comparison[cite: 1].
  * **Compares & Logs (Room 2):** Evaluates decoded Barcode/OCR strings against the master preset (`OK` or `NG`) and pushes scan records (`logs/scan_logsheet.csv`) and failure images (`captures/`) to a central traceability logsheet[cite: 1].
* **3-Tier Read Quality Verification (`Good` / `Not Good` / `Error`):** Grades read health using ML classification (`quality_classifier.pt`) and software sharpness/retry metrics to warn operators before complete read failures occur[cite: 1].
* **Direct `0 / 1` Digital Sensor I/O & Two-Tier Access Control:**
  * **Operator / User Tier (Unlocked):** Configures opto-isolated `0 / 1` GPIO triggers, `OK / NG` outputs, ROI boxes, and safe I/O logic rules[cite: 1].
  * **Dev / Supplier Tier (Locked):** Core Python/OpenCV/ML source code, OS files, and factory calibration parameters remain password/SSH-locked[cite: 1].

---

## Scope Boundaries

* **In Scope:** Ethernet camera ingestion (`RTSP`, `HTTP/MJPEG`, `WebRTC`), ML + ROI box detection, 3-bank software preprocessing, Barcode + OCR decoding, `Good / Not Good / Error` quality grading, Two-Room preset/logsheet sync, `0 / 1` GPIO I/O, and User vs. Dev role lockout[cite: 1].
* **Excluded (Out of Scope):** External PLC hardware and industrial fieldbuses (`PROFINET`, `EtherNet/IP`, `OPC UA`, `Modbus TCP`); custom camera sensors, motorized lenses, or integrated LED lighting hardware; formal `ISO/IEC 15415` optical grading; and embedded Lua scripting[cite: 1].
* **Optional Stretch Goal:** Master/Slave multi-head linking over Ethernet to synchronize triggers and consolidate scan results across multiple Raspberry Pi nodes[cite: 1].

---
