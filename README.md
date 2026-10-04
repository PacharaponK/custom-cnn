# custom-cnn

Custom compact U-Net สำหรับ lane-area segmentation วิชา 241-353 AI Ecosystem Assignment 10
รหัสนักศึกษา 6610110190 · single class · train ทุก layer from scratch

Input W×H=64×36 · train/test=743/400 · batch4 · 30epochs · 29,761 parameters
Test mean IoU **0.81213494** · detected IoU≥0.6 **400/400**

## สิ่งที่อยู่ใน Git

- Code เตรียม polygon, training, inference, evaluation, memory และ verification
- pyproject.toml, uv.lock และ .python-version สำหรับ environment ด้วย uv
- Manifest/split/exclusions/dedup records และ initial/final checkpoint ของ final run
- Metrics CSV/JSON, กราฟ/6 inference snapshots และ TensorBoard event file
- [Architecture](custom-unet-architecture.md) และ build_report.py สำหรับสร้าง HTML report ในเครื่อง

Dataset JPG/XML, prepared images/masks/NPZ caches, prediction masks, environments, local paths,
รายงาน HTML/Word/PDF, ZIP และ QA intermediates ไม่อยู่ใน Git; สร้างข้อมูลและ prediction ใหม่ด้วยขั้นตอนด้านล่าง

## Environment

Python 3.12.10; PyTorch CUDA12.8 wheel จาก official index ใน pyproject.toml

~~~powershell
git clone https://github.com/PacharaponK/custom-cnn.git
cd custom-cnn
uv sync --locked
uv run --locked python test_pipeline.py
uv run --locked python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
~~~

Inference/training รองรับ CPU fallback แต่ผล benchmark ในรายงานวัดบน RTX3050 Laptop GPU
ไม่ติดตั้ง packages ด้วย pip แยกจาก lockfile ของ uv

## เตรียม dataset ที่ไม่ได้แนบใน Git

ใช้ **dataset ต้นฉบับจากโจทย์เดียวกัน** จำนวน JPG1,200/XML9; วางใต้ source-dataset/
รักษา subfolder/basename จากชุดต้นฉบับ เพราะ manifest ใช้ relative paths ของ source
อีกแหล่งคือ source-dataset/ ใน ZIP ชุดส่งที่จัดไว้ในเครื่อง (ZIP ไม่ได้ upload กับ Git)

Repo นี้ยังไม่มี public dataset download URL; ผู้รันต้องมี dataset จากโจทย์ก่อนทำขั้นตอนนี้
config.json ใช้ source_dataset=source-dataset ตามค่าเริ่มต้น หรือกำหนด absolute path ไปยังชุดต้นฉบับ

data/ ใน Git เป็น **reference metadata เท่านั้น** จึงต้องย้ายเก็บไว้ก่อน prepare เพื่อไม่ทับ manifest เดิม

~~~powershell
Move-Item -LiteralPath data -Destination data-reference
uv run --locked python prepare_data.py
~~~

คำสั่ง Move-Item ใช้ครั้งเดียวหลัง clone และก่อน prepare. หาก data/ มี prepared assets แล้วไม่ต้องรันซ้ำ
prepare_data.py ปฏิเสธการทับ prepared manifest เพื่อรักษาผลเดิม

ขั้นตอน prepare:
- Deduplicate52 copies และ exclude400/401/451/452/544 ตาม policy ที่ยืนยัน
- ใช้เฉพาะ polygon label lane: **พื้นที่เลน** ไม่ใช่เส้นสีขาว; class0
- สร้าง YOLO normalized coordinates8 decimals และ original GT1280×720 แบบ union
- Clip raster points ที่ W−1/H−1; roundtrip reconstruction ถึง source0.01px
- Resize RGB bilinear/GT nearest-neighbor และสร้าง small NPZ caches
- Temporal split743/400 และ SHA-256 overlap0

Manifest ที่สร้างต้องตรง final run SHA-256:
2619adc2a95a3a18cca1bc16efa53881044c6f7cc09ecbc4fa9b08243edd39bf

load_model ตรวจ manifest hash และ config การทดลองทุก field
source_dataset/vault_assignment เป็นตำแหน่งไฟล์จึงเปลี่ยนได้เมื่อย้ายเครื่อง;
width/height/seed/threshold/augmentation/run ID และพารามิเตอร์อื่นยังต้องตรง checkpoint

## ทำ inference และตรวจผล final checkpoint

Prediction masks ไม่อยู่ใน Git จึงต้องสร้างก่อน verification

~~~powershell
uv run --locked python predict.py
uv run --locked python evaluation.py
uv run --locked python verify_results.py
~~~

predict.py ไม่ทับ existing prediction masks; หากเคยรันแล้วให้ใช้ verify_results.py เพื่อตรวจซ้ำ
Verification คำนวณ IoU จาก saved PNG400 ภาพ ตรวจ hashes/30epochs/split และ reload checkpoint
ทำนาย first/last test สองภาพเทียบ saved pixels. GPU/CPU หรือ runtime ต่างกันอาจให้ผล rounding ต่างกัน

## Training ใหม่

เปลี่ยน run_id ใน config.json เป็นชื่อใหม่ก่อน train. ไม่ใช้ test เพื่อเลือก architecture,
LR, probability threshold หรือ checkpoint. หากต้อง tune ให้แบ่ง validation จาก development images

~~~powershell
uv run --locked python train.py --smoke
uv run --locked python train.py
uv run --locked python predict.py
uv run --locked python evaluation.py
uv run --locked python measure_memory.py
~~~

Smoke ใช้8 train images100 updates; final runเริ่ม random initialization ใหม่ ไม่ใช้ smoke weights ต่อ
Final checkpoint ใช้ epoch30; BCEWithLogits, Adam LR.001, photometric augmentation train-only

## กราฟ Memory และรายงาน

~~~powershell
uv run --locked tensorboard --logdir runs --host 127.0.0.1
uv run --locked python measure_memory.py
~~~

Memory benchmark processแยก B1 FP32 warmup10/repetitions100; forward/E2E แยกกัน
GPU allocated/reserved ไม่รวม CUDA driver/context; CPU RSS sampled5ms คือทั้ง process
E2E รวม JPEG/resize/model/mask/PNG encoding ใน memory ไม่รวม disk write.
ค่าที่วัดใหม่ขึ้นกับ hardware และไม่ใช้แทนผลเดิมโดยไม่บันทึก run/provenance

รายงาน HTML/Word/PDF เก็บไว้ในเครื่องและถูก ignore. HTML ที่สร้างอ่านได้ offline เพราะฝังภาพไว้แล้ว. build_report.py ใช้ evidence ของ final run ที่แนบ
จึง rebuild HTML ของ final run ได้หลัง verification. รายงาน Word ใช้ TH Sarabun New เนื้อหา16pt
ตัวอักษรและตารางขาวดำ; ภาพผลทดลองคงสีเดิม. Scripts เฉพาะเครื่องสำหรับ Word/ZIP/export vault ไม่อยู่ใน Git
รายงานถูกสร้างก่อน Git publication; สถานะ publication ที่ปรากฏในรายงานเป็นประวัติ ณ เวลาสร้าง

## Evaluation และข้อจำกัด

Saved PNG: logits→sigmoid→bilinear probability upsample1280×720→threshold.5→0/255
IoU เป็น foreground intersection/union รายภาพ; main comparator≥.6 ตาม PDF และบันทึก>.6 ตาม TXT
Detection flag อาศัย GT หลัง inference ไม่ใช่ model confidence หรือ mAP
Empty-empty IoU=1 พร้อมนับ; ไม่มี detected image ให้ mean detected เป็น null
Report mean detected และ mean all พร้อม count/total; final runทั้งสองค่าเท่ากันเพราะทุกภาพผ่านเกณฑ์

Test มาจากช่วงท้ายของ sequence/route เดียว ผลนี้ไม่ยืนยัน generalization ทุกถนน/กล้อง/แสง/วันเก็บข้อมูล
Train micro IoU บน resized/augmented batches กับ test macro IoU ที่ original resolution เทียบตรง ๆ ไม่ได้
