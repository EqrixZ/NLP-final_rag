# screenshots/

ใส่ภาพหน้าจอ (PNG/JPG) ของแอปที่ deploy แล้วไว้ในโฟลเดอร์นี้ ตามรายการใน [`docs/screenshot_guide.md`](../docs/screenshot_guide.md)

1. ตั้งชื่อไฟล์ตามลำดับ เช่น `01_home.png`, `02_thai_question_sources.png`, ...
2. คัดลอก `captions.example.json` เป็น `captions.json` แล้วแก้คำอธิบายให้ตรงกับภาพจริง
3. รัน `python make_screenshot_pdf.py` → ได้ `screenshots.html` และ `screenshots.pdf` ที่โฟลเดอร์หลักของโปรเจกต์
