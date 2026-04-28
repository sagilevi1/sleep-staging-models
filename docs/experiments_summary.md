# סיכום תהליך הניסויים — Sleep Staging Models

מסמך זה מסכם את שלושת הניסויים שבוצעו עד כה בפרויקט סיווג שלבי שינה על דאטהסט DREAMT, כולל מטרות, ארכיטקטורה, שינויים בין הניסויים, ותוצאות בפועל.

---

## ניסוי 1 — Baseline / Triple Stream Single-Window

### מטרה
לבדוק את הארכיטקטורה הקיימת כפי שהיא: מודל שמקבל חלון יחיד של 30 שניות ומסווג אותו לאחד משלבי השינה. המטרה הייתה להבין האם המודל מצליח ללמוד את הדאטה, והאם הבעיה היא חוסר יכולת למידה או בעיית הכללה.

### ארכיטקטורה
מודל Triple Stream:

```
BVP / PPG window  → encoder
ACC window        → encoder
IBI / HRV features → encoder
                   ↓
              fusion layer
                   ↓
              classifier
                   ↓
          sleep stage prediction
```

כל דוגמה: `30-second window → one sleep-stage label`. המודל לא רואה הקשר טמפורלי לפני או אחרי החלון.

### תוצאות ולקחים
תבנית ברורה של **overfitting חמור**:
- Train accuracy עלה מאוד
- Validation accuracy נשאר תקוע
- Validation loss המשיך לעלות
- Validation kappa נשאר נמוך ויציב

המודל מספיק חזק כדי לשנן את סט האימון אך אינו מכליל ל-subjects חדשים.

### מסקנה מרכזית
הבעיה: המודל מתייחס ל-sleep staging כבעיה של window בודד, אבל בפועל זו בעיה טמפורלית — שלב שינה תלוי בהקשר (חלונות קודמים, מעברים בין N2/N3, מחזורי REM, התעוררויות קצרות).

---

## ניסוי 2 — Triple Stream + Regularization + Augmentations + Focal Loss

### מטרה
לבדוק האם אפשר לשפר את אותה ארכיטקטורה בלי לשנות את מבנה הבעיה. השאלה: האם הבעיה היא בעיקר חוסר regularization, או עמוקה יותר ודורשת context טמפורלי?

### שינויים לעומת ניסוי 1
| שינוי | ערך | מטרה |
|---|---|---|
| Dropout | 0.2 → 0.4 | הקטנת שינון |
| Augmentations | Gaussian noise, time shift, modality dropout, amplitude jitter | מניעת שינון windows ספציפיים |
| Loss | CrossEntropy → Focal Loss (γ=2.0, weighted) | משקל לדוגמאות קשות (N1/REM/N3) |
| Weight decay | 1e-5 → 1e-4 | regularization על משקלים |
| Scheduler | ReduceLROnPlateau patience קצר יותר | הורדת LR מהירה כשאין שיפור |

### תוצאות בפועל (Run ID: `2026-04-25_002_triple_v2_focal_aug`)

**Validation:**
- Best Val Kappa: **0.2539** (epoch 15)
- Best Val F1: 0.4290
- Early stopping בארפוק 30

**Test:**
- Test Kappa: **0.2862**
- Test Accuracy: 0.4531
- Test F1 (weighted): 0.4548
- Test F1 (macro): 0.3253

**Per-class (test):**
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| P (Wake-pre) | 0.61 | 0.81 | 0.70 | 4383 |
| W | 0.27 | 0.22 | 0.24 | 2656 |
| N1 | 0.16 | 0.22 | 0.18 | 1378 |
| N2 | 0.63 | 0.42 | 0.50 | 6832 |
| N3 | 0.08 | 0.15 | 0.10 | 375 |
| REM | 0.20 | 0.27 | 0.23 | 1352 |

### תובנות
- שיפור מתון על ה-baseline אך עדיין נמוך משמעותית מ-state-of-the-art ב-sleep staging.
- Train loss המשיך לרדת (0.44 בסוף) בעוד Val loss טיפס (3.5+) — overfitting עדיין משמעותי גם עם regularization.
- מחלקות מיעוט (N3, N1, REM) עדיין סובלות מביצועים חלשים.
- המסקנה מתחזקת: regularization לבד לא פותר את הבעיה — חסר context טמפורלי.

---

## ניסוי 3 — Sequence Model with BiLSTM

### מטרה
שינוי ארכיטקטוני: מעבר מסיווג חלון בודד לסיווג רצף של חלונות. ההשערה: הוספת context טמפורלי (≈10 דקות) תפתור את תקרת הביצועים.

### ארכיטקטורה
```
20 consecutive windows → shared Triple Stream encoder per window
                       ↓
                    BiLSTM
                       ↓
              per-window classifier (20 predictions)
```

- Sequence length: 20 windows (10 דקות)
- Stride: 5
- רצפים לא חוצים בין subjects
- חלוקה: 14,386 train / 3,111 val / 3,342 test sequences

### תוצאות בפועל (Run ID: `2026-04-25_001_sequence_v1_bilstm`)

**הריצה נכשלה — NaN losses מאפוק 1:**
```
Epoch 1: Train Loss=nan, Train Acc=0.4462, Val Loss=nan, Val Kappa=0.0000
Epoch 2-15: כל האפוקים עם nan loss, Val Kappa=0
Early stopping at epoch 15
Test Kappa: 0.0000
```

המודל לא למד כלום — ה-loss הפך NaN כבר בשלב מוקדם של אפוק 1, וה-LR scheduler רק הוריד את ה-LR בלי לפתור את הבעיה.

### חשד לסיבות (לחקירה)
1. **בעיית NaN ב-encoder או ב-BiLSTM**: ייתכן שהמשקלים מתפוצצים, או שהמיזוג בין mixed precision (AMP) לבין BiLSTM/weight_norm יוצר NaN.
2. **Initialization**: 22.4M פרמטרים — אולי דרוש gradient clipping או init יציב יותר.
3. **AMP/autocast**: ידוע כיוצר NaN במקרי קצה עם RNNs — שווה לנסות `fp32` או `gradient scaler` עם `init_scale` נמוך יותר.
4. **Input scaling**: ייתכן שיש NaN/Inf בקלט (modality dropout, normalization) שלא טופלו.
5. **Loss**: focal loss על per-window predictions ברצף — אולי בעיה במסכה/padding.

### צעדים מומלצים לריצת המשך
- להוסיף `torch.nn.utils.clip_grad_norm_(..., 1.0)` ב-training step.
- להריץ ראשית בלי AMP (`use_amp=False`) כדי לבודד את הגורם.
- לבדוק `torch.isnan` על קלט ועל logits באפוק הראשון.
- להוריד LR התחלתי מ-1e-4 ל-3e-5 עד שהאימון יציב.
- לוודא שאין NaN בערכי IBI/HRV (רגישים לערכים חסרים).

---

## טבלת השוואה כוללת

| ניסוי | Best Val Kappa | Test Kappa | Test Acc | סטטוס |
|---|---|---|---|---|
| 1 — Baseline | נמוך, overfitting | — | — | baseline |
| 2 — Focal+Aug | 0.2539 | **0.2862** | 0.4531 | הצליח, שיפור מתון |
| 3 — BiLSTM | 0.0000 | 0.0000 | — | **כשל (NaN)** — לדבג |

---

## מערכת התיעוד (ExperimentTracker)

נוספה תשתית tracking שמתעדת לכל ריצה:
- Run ID, git commit, branch, dirty flag
- Config + hyperparameters
- Dataset split + class distributions
- Model parameters count
- Training curves
- Best epoch + checkpoint
- Confusion matrix + classification report
- Test metrics
- Environment

המטרה: יכולת מלאה לשחזר כל ריצה — לדעת בדיוק איזה commit, config, preprocessing, hyperparameters, split, loss, augmentation ו-checkpoint יצרו כל תוצאה.

---

## משפט מסכם

הריצה הראשונה הראתה שהמודל מסוגל ללמוד את סט האימון אך אינו מכליל היטב, מה שהצביע על overfitting ועל מגבלה בסיסית של גישת single-window. הריצה השנייה (Focal + augmentations + regularization) הניבה שיפור מתון בלבד (Test Kappa = 0.286), המאשר שהבעיה אינה רק regularization. הריצה השלישית (BiLSTM על רצפים של 20 חלונות) נכשלה טכנית בגלל NaN loss, ולכן ההשערה המרכזית — שהוספת context טמפורלי תיתן קפיצת ביצועים — **טרם נבחנה בפועל** ודורשת ריצה חוזרת אחרי תיקון יציבות האימון.
