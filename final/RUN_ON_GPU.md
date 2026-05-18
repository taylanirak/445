# GPU'da en iyi sonucu almak — adımlar (arkadaşlar için)

Bu klasör kendi kendine yeter (kod + `scoring.py` + `data/`). GPU'lu makinede
**kod hiç değiştirilmeden** çalıştırılır; `run_all.py` GPU'yu otomatik algılar
ve **kanonik konfigürasyonu** seçer:

> DeBERTa-v3-large + DeBERTa-v3-large-MNLI · 3 seed · 4 epoch · 2 NLI şablonu ·
> CORN ablasyonu dahil — yani alınabilecek en iyi sonuç.

## A) Sade terminal (en garantili)

Zip'i bir klasöre açın. O klasörde (içinde `final/`, `scoring.py`, `data/` görünür):

```bash
# 1) CUDA'lı PyTorch + bağımlılıklar (CUDA sürümünü makinenize göre seçin)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r final/requirements_final.txt

# 2) Tüm sistemi eğit + resmi scoring.py ile skorla  (GPU'da ~2–3.5 saat)
python final/src/run_all.py

# 3) Rapor + slayt + çalıştırılmış notebook + figürler + gönderim + zip
python final/_finalize.py
```

Bitince:
- Skorlar: `final/results.json` (ve konsolda "ENSEMBLE dev (official): ...").
- Teslim arşivi: `final/GroupXX_final_submission.zip`.
- CodaBench dosyası: `final/predictions/ensemble_test.jsonl`.

Çökerse tekrar `python final/src/run_all.py` çalıştırın — bileşenler
`final/.cache/`'e yazıldığı için kaldığı yerden devam eder.

## B) Codex / Claude Code ajanına verilecek PROMPT

Aşağıdaki metni olduğu gibi yapıştırın:

> This repo already contains a complete, working SemEval-2026 Task 5 system in
> `final/`. Do NOT change the methodology, models, or scoring. A GPU is
> available. Your only job:
> 1. Install deps: a CUDA build of `torch`, then `pip install -r requirements.txt`
>    and `pip install -r final/requirements_final.txt`.
> 2. Run `python final/src/run_all.py` from the repo root. It auto-detects the
>    GPU and runs the canonical config (DeBERTa-v3-large, 3 seeds, 4 epochs,
>    2 NLI templates, CORN). If it stops, just run it again — it resumes from
>    `final/.cache/`.
> 3. Run `python final/_finalize.py` to rebuild the report (pdf/docx), slides,
>    the executed notebook (keep outputs), figures, validate the submission and
>    build the archive.
> 4. Report the official numbers from `final/results.json`
>    (`ensemble.dev_continuous` and `ensemble.verdict`) and confirm
>    `final/GroupXX_final_submission.zip` was rebuilt. Do not fabricate numbers.

## C) Teslimden önce (1 dakikalık, puan kaybı önler)

Aşağıdaki yer tutucuları gerçek değerlerle değiştirin (sonra `_finalize.py`'yi
tekrar çalıştırın ki rapor/slayt güncellensin):

- `final/GroupXX_final_report.md` içindeki `GroupXX`, `Member 1/2/3`, tarih.
- Dosya adlarındaki `GroupXX_` ön ekini gerçek grup numaranızla değiştirmek
  isterseniz: dosyaları yeniden adlandırın **ve** `final/_make_archive.py`
  içindeki `GroupXX` adlarını güncelleyin, sonra `python final/_make_archive.py`.

## Beklenti (dürüst)

CPU'da en iyi: acc@std ≈ 0.65 / Spearman ≈ 0.37 ("Below OK").
GPU kanonik konfigde aynı mimariyle acc@std ≈ 0.70–0.80 / ρ ≈ 0.5–0.7
beklenir → büyük olasılıkla **OK, şansa göre Good**. Garanti değil (veride
annotatör gürültü tavanı var) ama GPU bunu güvenle yukarı çeker.
