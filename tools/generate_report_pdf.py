"""
generate_report_pdf.py
Generates reports/project_report.pdf using ReportLab.
Run from the project root:
    python tools/generate_report_pdf.py
"""

import os, sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.colors import HexColor, white, black

# ── Output path ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PATH  = os.path.join(PROJECT_ROOT, "reports", "project_report.pdf")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
C_PRIMARY   = HexColor("#2563eb")
C_ACCENT    = HexColor("#7c3aed")
C_SUCCESS   = HexColor("#059669")
C_WARNING   = HexColor("#d97706")
C_DANGER    = HexColor("#dc2626")
C_BG_BLUE   = HexColor("#eff6ff")
C_BG_GREEN  = HexColor("#f0fdf4")
C_BG_AMBER  = HexColor("#fffbeb")
C_BG_RED    = HexColor("#fef2f2")
C_LIGHT     = HexColor("#f1f5f9")
C_BORDER    = HexColor("#e2e8f0")
C_MUTED     = HexColor("#64748b")
C_DARK      = HexColor("#1e293b")
C_COVER_BG  = HexColor("#0f172a")

W, H = A4   # 595.27 x 841.89 pts

# ── Page setup ─────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT_PATH,
    pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2.5*cm,  bottomMargin=2.5*cm,
    title="MeMOTR Vehicle Tracking Enhancement — Project Report",
    author="MajorProject-MeMOTR",
    subject="Phase 5/6 Interim Results",
)

CONTENT_WIDTH = W - 4.4*cm   # usable width

# ── Styles ─────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=base[parent], **kw)

styles = {
    "cover_title": S("cover_title", fontSize=34, textColor=white,
                     fontName="Helvetica-Bold", leading=42, alignment=TA_CENTER),
    "cover_sub":   S("cover_sub",   fontSize=14, textColor=HexColor("#94a3b8"),
                     fontName="Helvetica",       leading=22, alignment=TA_CENTER),
    "cover_meta":  S("cover_meta",  fontSize=11, textColor=HexColor("#64748b"),
                     fontName="Helvetica",       alignment=TA_CENTER),

    "h1": S("h1", fontSize=22, textColor=C_PRIMARY, fontName="Helvetica-Bold",
            leading=28, spaceBefore=16, spaceAfter=6),
    "h2": S("h2", fontSize=16, textColor=C_DARK,    fontName="Helvetica-Bold",
            leading=22, spaceBefore=14, spaceAfter=4),
    "h3": S("h3", fontSize=13, textColor=C_DARK,    fontName="Helvetica-Bold",
            leading=18, spaceBefore=10, spaceAfter=3),

    "body": S("body", fontSize=10, textColor=C_DARK, fontName="Helvetica",
              leading=15, spaceBefore=2, spaceAfter=4, alignment=TA_JUSTIFY),
    "body_sm": S("body_sm", fontSize=9, textColor=C_MUTED, fontName="Helvetica",
                 leading=13, spaceBefore=1, spaceAfter=2),

    "bullet": S("bullet", fontSize=10, textColor=C_DARK, fontName="Helvetica",
                leading=15, leftIndent=16, spaceBefore=1, spaceAfter=1,
                bulletIndent=6),

    "code": S("code", fontSize=8.5, textColor=HexColor("#1e40af"),
              fontName="Courier", leading=12, backColor=HexColor("#f1f5f9"),
              borderPadding=(2,4,2,4)),

    "callout_title": S("callout_title", fontSize=8, textColor=C_MUTED,
                       fontName="Helvetica-Bold", leading=12,
                       textTransform="uppercase", spaceAfter=3),
    "callout_body":  S("callout_body",  fontSize=9, textColor=C_DARK,
                       fontName="Helvetica", leading=13),

    "section_num": S("section_num", fontSize=11, textColor=white,
                     fontName="Helvetica-Bold", leading=14, alignment=TA_CENTER),

    "toc": S("toc", fontSize=10, textColor=C_DARK, fontName="Helvetica",
             leading=18, leftIndent=20),

    "delta_pos": S("delta_pos", fontSize=9, textColor=C_SUCCESS,
                   fontName="Helvetica-Bold", alignment=TA_CENTER),
    "delta_neg": S("delta_neg", fontSize=9, textColor=C_DANGER,
                   fontName="Helvetica-Bold", alignment=TA_CENTER),
    "delta_neu": S("delta_neu", fontSize=9, textColor=C_MUTED,
                   fontName="Helvetica",     alignment=TA_CENTER),
    "td":        S("td",        fontSize=9,  textColor=C_DARK,
                   fontName="Helvetica",     leading=13),
    "th":        S("th",        fontSize=8,  textColor=C_MUTED,
                   fontName="Helvetica-Bold",leading=11,
                   alignment=TA_LEFT, textTransform="uppercase"),
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def P(text, style="body"):
    if isinstance(style, str):
        return Paragraph(text, styles[style])
    return Paragraph(text, style)   # ParagraphStyle object passed directly

def B(text): return P(f"• &nbsp;{text}", "bullet")

def HR(color=C_BORDER, thickness=1):
    return HRFlowable(width="100%", thickness=thickness, color=color,
                      spaceBefore=4, spaceAfter=4)

def SP(h=0.25): return Spacer(1, h*cm)

def section_heading(num, title):
    """Coloured number chip + heading side-by-side."""
    chip = Table([[P(str(num), "section_num")]], colWidths=[0.75*cm],
                 rowHeights=[0.75*cm])
    chip.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(0,0), C_PRIMARY),
        ("ROUNDEDCORNERS", [6]),
        ("VALIGN",     (0,0),(0,0), "MIDDLE"),
    ]))
    row = Table([[chip, P(title, "h1")]], colWidths=[1.0*cm, CONTENT_WIDTH-1.0*cm])
    row.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                             ("TOPPADDING",(0,0),(-1,-1),0),
                             ("BOTTOMPADDING",(0,0),(-1,-1),0)]))
    return [row, HR(C_PRIMARY, thickness=1.5), SP(0.15)]

def callout(text, title=None, bg=C_BG_BLUE, border=C_PRIMARY):
    content = []
    if title:
        content.append(P(f"<b>{title}</b>", "callout_title"))
    content.append(P(text, "callout_body"))
    inner = Table([[c] for c in content], colWidths=[CONTENT_WIDTH - 1.2*cm])
    inner.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),2),
                                ("BOTTOMPADDING",(0,0),(-1,-1),2)]))
    outer = Table([[inner]], colWidths=[CONTENT_WIDTH])
    outer.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(0,0), bg),
        ("LEFTPADDING",  (0,0),(0,0), 10),
        ("RIGHTPADDING", (0,0),(0,0), 8),
        ("TOPPADDING",   (0,0),(0,0), 8),
        ("BOTTOMPADDING",(0,0),(0,0), 8),
        ("LINEBEFORE",   (0,0),(0,0), 4, border),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return [outer, SP(0.2)]

def data_table(headers, rows, col_widths=None, highlight_rows=None):
    highlight_rows = highlight_rows or []
    th_style = [P(h, "th") for h in headers]
    data = [th_style]
    for row in rows:
        data.append([P(str(c), "td") if isinstance(c, str) else c for c in row])

    if col_widths is None:
        col_widths = [CONTENT_WIDTH / len(headers)] * len(headers)

    t = Table(data, colWidths=col_widths, repeatRows=1)
    ts = [
        ("BACKGROUND",    (0,0),(-1,0),  C_LIGHT),
        ("LINEBELOW",     (0,0),(-1,0),  1.5, C_BORDER),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, HexColor("#f8fafc")]),
        ("LINEBELOW",     (0,1),(-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 8),
        ("RIGHTPADDING",  (0,0),(-1,-1), 8),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ("BOX",           (0,0),(-1,-1), 1,   C_BORDER),
    ]
    for r in highlight_rows:
        ts.append(("BACKGROUND", (0, r+1), (-1, r+1), C_BG_BLUE))
        ts.append(("FONTNAME",   (0, r+1), (-1, r+1), "Helvetica-Bold"))
    t.setStyle(TableStyle(ts))
    return [t, SP(0.3)]

def stat_cards(items):
    """items = list of (value, label) tuples"""
    cells = []
    for val, label in items:
        inner = Table([
            [P(str(val), S("sv", fontSize=20, textColor=C_PRIMARY,
                           fontName="Helvetica-Bold", leading=24, alignment=TA_CENTER))],
            [P(label,    S("sl", fontSize=8,  textColor=C_MUTED,
                           fontName="Helvetica", leading=10, alignment=TA_CENTER))],
        ])
        inner.setStyle(TableStyle([("TOPPADDING",(0,0),(-1,-1),3),
                                    ("BOTTOMPADDING",(0,0),(-1,-1),3)]))
        cell = Table([[inner]])
        cell.setStyle(TableStyle([
            ("BOX",            (0,0),(0,0), 1, C_BORDER),
            ("BACKGROUND",     (0,0),(0,0), white),
            ("TOPPADDING",     (0,0),(0,0), 10),
            ("BOTTOMPADDING",  (0,0),(0,0), 10),
            ("LEFTPADDING",    (0,0),(0,0), 8),
            ("RIGHTPADDING",   (0,0),(0,0), 8),
            ("LINEABOVE",      (0,0),(0,0), 3, C_PRIMARY),
        ]))
        cells.append(cell)

    n = len(items)
    w = CONTENT_WIDTH / n - 0.2*cm
    row = Table([cells], colWidths=[w]*n)
    row.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),3),
                              ("RIGHTPADDING",(0,0),(-1,-1),3)]))
    return [row, SP(0.3)]

# ── Cover page ─────────────────────────────────────────────────────────────────
def cover_page():
    items = []

    # Dark background block
    cover_data = [
        [P("MAJOR PROJECT TECHNICAL REPORT", S("badge", fontSize=8,
           textColor=HexColor("#93c5fd"), fontName="Helvetica-Bold",
           alignment=TA_CENTER, spaceAfter=16))],
        [SP(0.5)],
        [P("MeMOTR Vehicle Tracking<br/>Enhancement", "cover_title")],
        [SP(0.3)],
        [P("Scene-Adaptive Detection Threshold (C2) and Perspective-Weighted<br/>"
           "Loss (C1) for Multi-Object Tracking in Adverse Conditions", "cover_sub")],
        [SP(0.8)],
        [HR(HexColor("#334155"))],
        [SP(0.4)],
        [P("UA-DETRAC · VisDrone · IOTD (Pending)  |  "
           "Phases 1–6 Interim  |  September 2026", "cover_meta")],
        [SP(0.3)],
        [Table(
            [[P("47", S("cm_val", fontSize=28, textColor=white,
                        fontName="Helvetica-Bold", alignment=TA_CENTER)),
              P("62.13", S("cm_val", fontSize=28, textColor=white,
                           fontName="Helvetica-Bold", alignment=TA_CENTER)),
              P("5+5", S("cm_val",  fontSize=28, textColor=white,
                          fontName="Helvetica-Bold", alignment=TA_CENTER)),
              P("209", S("cm_val",  fontSize=28, textColor=white,
                          fontName="Helvetica-Bold", alignment=TA_CENTER))],
             [P("Test Sequences", S("cm_lbl", fontSize=9, textColor=HexColor("#64748b"),
                                    fontName="Helvetica", alignment=TA_CENTER)),
              P("Baseline HOTA",  S("cm_lbl", fontSize=9, textColor=HexColor("#64748b"),
                                    fontName="Helvetica", alignment=TA_CENTER)),
              P("Training Epochs",S("cm_lbl", fontSize=9, textColor=HexColor("#64748b"),
                                    fontName="Helvetica", alignment=TA_CENTER)),
              P("C2 Params",      S("cm_lbl", fontSize=9, textColor=HexColor("#64748b"),
                                    fontName="Helvetica", alignment=TA_CENTER))]],
            colWidths=[(CONTENT_WIDTH-1.6*cm)/4]*4)],
    ]

    cover_tbl = Table([[row[0]] for row in cover_data], colWidths=[CONTENT_WIDTH])
    cover_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_COVER_BG),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 32),
        ("RIGHTPADDING",  (0,0),(-1,-1), 32),
        ("ROUNDEDCORNERS", [8]),
    ]))

    items.append(cover_tbl)
    items.append(PageBreak())
    return items

# ── Build story ────────────────────────────────────────────────────────────────
story = []
story += cover_page()

# ── 1. Project Overview ────────────────────────────────────────────────────────
story += section_heading(1, "Project Overview")
story += stat_cards([
    ("MeMOTR", "Base architecture — end-to-end transformer MOT"),
    ("C1 + C2", "Two novel contributions to MeMOTR"),
    ("50.4 M", "Frozen backbone parameters"),
    ("209", "Trainable ThresholdAdapter params (C2)"),
])
story.append(P(
    "This project extends <b>MeMOTR</b> (Memory-augmented Transformer for Multi-Object Tracking) "
    "with two targeted contributions designed to improve vehicle detection and tracking in "
    "<b>adverse lighting conditions</b>, specifically the night-time and dusk scenarios present "
    "in the Indian On-road Traffic Dataset (IOTD)."))
story.append(P(
    "MeMOTR uses track-query embeddings that propagate across frames, giving it strong long-term "
    "association capability. However, its fixed detection threshold (τ = 0.5) is ill-suited for "
    "scenes where genuine vehicles score 0.35–0.45 (dark/low-contrast). "
    "The MeMOTR backbone and encoder are <b>not modified</b> — all contributions attach as "
    "lightweight modules, preserving pretrained generalisation."))
story += callout(
    "All contributions operate behind feature flags (<code>use_c1</code>, <code>use_c2</code>) "
    "defaulting to False, ensuring zero regression for the unmodified baseline.",
    title="Design Constraint", bg=C_BG_BLUE, border=C_PRIMARY)

# ── 2. Problem Statement ───────────────────────────────────────────────────────
story += section_heading(2, "Problem Statement")
story.append(P("<b>Issue 1 — Fixed Detection Threshold (→ C2)</b>", "h3"))
story.append(P(
    "MeMOTR's runtime tracker admits new detections only when classification score ≥ τ = 0.5. "
    "In dark/dusk IOTD scenes, genuine vehicles frequently score 0.35–0.45, causing silent "
    "false-negatives. In bright overexposed conditions, the same threshold admits background clutter."))
story.append(P("<b>Issue 2 — Perspective-Blind Training Loss (→ C1)</b>", "h3"))
story.append(P(
    "Uniform loss weight λ across all frame regions fails to account for perspective geometry. "
    "In overhead footage, vehicles at varying distances appear at very different scales; "
    "a perspective-weighted λ(x,y) would better calibrate the model for multi-scale scenarios."))

# ── 3. Contributions ──────────────────────────────────────────────────────────
story += section_heading(3, "Novel Contributions")

story.append(P("<b>C2 — ThresholdAdapter (Scene-Adaptive Detection Threshold)</b>", "h2"))
story.append(P("Status: &nbsp;<font color='#059669'>✅ Implemented and Evaluated</font>"))
story.append(SP(0.15))
story.append(P(
    "<b>Hypothesis:</b> The detection admission threshold τ should adapt per-frame to scene "
    "conditions. In dark scenes a lower τ recovers genuine vehicles; in bright/cluttered scenes "
    "a higher τ suppresses false positives."))
story.append(P("<b>SceneAnalyzer</b> (analytic, 0 learnable params) computes a 3-D feature vector:", "h3"))
story += [B("Brightness — mean pixel intensity [0, 1]"),
          B("Contrast — standard deviation of intensity [0, 1]"),
          B("Track density — active track count / max_tracks [0, 1]")]
story.append(P("<b>ThresholdAdapter</b> — MLP 3 → 16 → 8 → 1, then sigmoid-and-scale:", "h3"))
story.append(P(
    "τ = τ_min + (τ_max − τ_min) × sigmoid(MLP(scene_vec)) = 0.3 + 0.4 × sigmoid(…) ∈ [0.3, 0.7]",
    "code"))
story.append(SP(0.1))
story.append(P(
    "Total trainable parameters: <b>209</b>. MeMOTR backbone remains fully frozen (0 of 50.4 M "
    "params trainable) throughout C2 training. Training supervision uses brightness-based proxy "
    "labels (MSE loss between predicted τ and brightness-derived target τ)."))

story.append(SP(0.2))
story.append(P("<b>C1 — PositionMLP (Perspective-Weighted Training Loss)</b>", "h2"))
story.append(P("Status: &nbsp;<font color='#d97706'>⏳ Architecture ready — training deferred pending IOTD annotation</font>"))
story.append(SP(0.15))
story.append(P(
    "A small MLP that maps each ground-truth box's (x, y) image position to a per-box loss "
    "weight λ(x, y). Boxes near the camera (typically near the image bottom) receive higher λ. "
    "Boxes near the horizon receive lower λ. Requires real perspective calibration from IOTD."))

# ── 4. Datasets ───────────────────────────────────────────────────────────────
story += section_heading(4, "Datasets")
story += data_table(
    ["Dataset", "Split", "Sequences", "Format", "Status"],
    [
        ["UA-DETRAC",       "train + test", "40 test",    "MOT Challenge",    "✅ Active"],
        ["VisDrone Vehicles","train + val",  "7 val",      "VisDrone (filtered)","✅ Active"],
        ["IOTD",            "all",          "TBD (≥3)",   "Custom",           "⏳ Pending"],
    ],
    col_widths=[3.5*cm, 2.5*cm, 2.5*cm, 4.5*cm, 2.5*cm]
)
story.append(P(
    "<b>VehicleMOT</b> — a custom unified dataset class was built to handle mixed-format image "
    "naming (UA-DETRAC uses <code>img%05d.jpg</code>, VisDrone uses <code>%07d.jpg</code>) "
    "via auto-detection, without renaming files. Designed with sequence-level splits only, "
    "pickle-safe module-level factory (Windows spawn), batch size 1, num_workers=0."))

# ── 5. Implementation Phases ──────────────────────────────────────────────────
story += section_heading(5, "Implementation Phases")
phases = [
    ("1", "Environment Setup & Baseline Verification", "✅ Complete",
     "Compiled MSDA CUDA op; patched TrackEval Python 3.9 issues; "
     "smoke test zero-regression verified; hardcoded paths fixed; git submodules added."),
    ("2", "Data Pipeline", "✅ Complete",
     "VehicleMOT dataset class; UA-DETRAC XML→MOT conversion; "
     "VisDrone vehicle category filtering; seqinfo.ini generation for all 47 sequences."),
    ("3", "Module Development", "✅ Complete",
     "SceneAnalyzer, ThresholdAdapter (C2) implemented and wired into RuntimeTracker "
     "behind use_c2=False flag. PositionMLP (C1) architecture designed."),
    ("4", "IOTD Integration", "⏳ Deferred",
     "Requires ≥3 annotated IOTD sequences. Perspective calibration for C1 blocked on this."),
    ("5", "Interim Baseline Training", "✅ Complete",
     "5 epochs on UA-DETRAC+VisDrone; loss 21.4→2.2; checkpoints 0–4 saved; "
     "baseline HOTA = 62.13."),
    ("6", "C2 Training & Interim Evaluation", "✅ Complete (Interim)",
     "ThresholdAdapter trained 5 epochs (MSE 0.001465→0.000687); "
     "C2 HOTA = 60.42; clamp verified; failure mode diagnosed."),
    ("6+", "Full C1+C2 Training", "🔜 Planned",
     "After IOTD annotation: joint C1+C2 training with real night calibration."),
]

phase_data = [["#", "Phase", "Status", "Key Deliverables"]]
for num, name, status, desc in phases:
    phase_data.append([
        P(num, S("pn", fontSize=9, textColor=C_PRIMARY, fontName="Helvetica-Bold",
                 alignment=TA_CENTER)),
        P(name, S("pname", fontSize=9, textColor=C_DARK, fontName="Helvetica-Bold")),
        P(status, S("pstat", fontSize=8, textColor=C_MUTED, fontName="Helvetica")),
        P(desc, S("pdesc", fontSize=8.5, textColor=C_DARK, fontName="Helvetica", leading=12)),
    ])

pt = Table(phase_data, colWidths=[0.8*cm, 4.2*cm, 3.0*cm, CONTENT_WIDTH-8.0*cm], repeatRows=1)
pt.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0),  C_LIGHT),
    ("LINEBELOW",     (0,0),(-1,0),  1.5, C_BORDER),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, HexColor("#f8fafc")]),
    ("LINEBELOW",     (0,1),(-1,-1), 0.5, C_BORDER),
    ("BOX",           (0,0),(-1,-1), 1,   C_BORDER),
    ("TOPPADDING",    (0,0),(-1,-1), 7),
    ("BOTTOMPADDING", (0,0),(-1,-1), 7),
    ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    # Colour status cells
    ("TEXTCOLOR", (2,2),(2,2), C_SUCCESS),
    ("TEXTCOLOR", (2,3),(2,3), C_SUCCESS),
    ("TEXTCOLOR", (2,4),(2,4), C_SUCCESS),
    ("TEXTCOLOR", (2,5),(2,5), C_WARNING),
    ("TEXTCOLOR", (2,6),(2,6), C_SUCCESS),
    ("TEXTCOLOR", (2,7),(2,7), C_SUCCESS),
    ("TEXTCOLOR", (2,8),(2,8), C_MUTED),
]))
story += [pt, SP(0.3)]

# ── 6. Phase 5 Training ───────────────────────────────────────────────────────
story += section_heading(6, "Phase 5 — Baseline Training")
story += stat_cards([
    ("5 epochs", "Training duration"),
    ("2e-5",     "Learning rate"),
    ("21.4→2.2", "Loss trajectory"),
    ("0",        "Backbone params trained (frozen)"),
])
story += data_table(
    ["Epoch", "Avg Loss", "Checkpoint"],
    [["0","~21.4","checkpoint_0.pth"],
     ["1","~8.1", "checkpoint_1.pth"],
     ["2","~4.8", "checkpoint_2.pth"],
     ["3","~3.1", "checkpoint_3.pth"],
     ["4 (eval)","~2.2","checkpoint_4.pth"],],
    col_widths=[2.5*cm, 4*cm, CONTENT_WIDTH-6.5*cm],
    highlight_rows=[4]
)

# ── 7. Phase 6 C2 Training ────────────────────────────────────────────────────
story += section_heading(7, "Phase 6 — ThresholdAdapter (C2) Training")
story += stat_cards([
    ("209",     "Trainable parameters"),
    ("50.4 M",  "Frozen parameters"),
    ("0.000687","Final MSE loss (epoch 4)"),
    ("~24 min", "Total training time"),
])
story += data_table(
    ["Epoch", "Avg MSE Loss", "Duration", "Checkpoint"],
    [["0","0.001465","5.4 min","c2_adapter_epoch_0.pth"],
     ["1","0.000964","4.7 min","c2_adapter_epoch_1.pth"],
     ["2","0.000947","4.6 min","c2_adapter_epoch_2.pth"],
     ["3","0.000852","4.6 min","c2_adapter_epoch_3.pth"],
     ["4","0.000687","4.6 min","c2_adapter_epoch_4.pth"],],
    highlight_rows=[4]
)

story.append(P("<b>Clamp Verification — uav0000117_02622_v (30 frames probed):</b>", "h3"))
story += data_table(
    ["Parameter", "Value", "Result"],
    [["tau_min clamp",     "0.300",    "Design parameter"],
     ["tau_max clamp",     "0.700",    "Design parameter"],
     ["Observed tau min",  "0.338972", "✅ Within range"],
     ["Observed tau max",  "0.344031", "✅ Within range"],
     ["Out-of-range",      "0 / 30",   "✅ Clamp working correctly"],
     ["Mean brightness",   "0.1805",   "Very dark — tau correctly near minimum"],],
    col_widths=[4.5*cm, 4*cm, CONTENT_WIDTH-8.5*cm]
)
story += callout(
    "The 61k-detection explosion on uav0000117 is NOT a clamp bug. "
    "Tau stayed within [0.3, 0.7] throughout. The issue is domain shift: "
    "at τ≈0.34, UAV-altitude footage admits nearly all 300 det-queries per frame "
    "because MeMOTR was never trained at this altitude/scale. "
    "Fix: IOTD calibration in Phase 6 proper.",
    title="✅ Verified Finding", bg=C_BG_GREEN, border=C_SUCCESS)

# ── 8. Evaluation Results ─────────────────────────────────────────────────────
story.append(PageBreak())
story += section_heading(8, "Evaluation Results")
story += callout(
    "INTERIM RESULTS — UA-DETRAC + VisDrone only. No IOTD data. "
    "These numbers will be superseded after Phase 4 IOTD annotation.",
    title="⚠ Interim Status", bg=C_BG_AMBER, border=C_WARNING)

def delta_p(text, positive=None):
    if positive is True:  return P(text, "delta_pos")
    if positive is False: return P(text, "delta_neg")
    return P(text, "delta_neu")

results_data = [
    ["Metric", "Baseline", "C2 Interim", "Δ (C2−BL)", "Notes"],
    ["HOTA ↑",       "62.13", "60.42", delta_p("−1.71", False), "FP surge in uav0000117"],
    ["DetA ↑",       "55.88", "52.91", delta_p("−2.97", False), "FP > FN reduction"],
    ["AssA ↑",       "69.45", "69.35", delta_p("−0.10"),        "✅ C2 doesn't touch association"],
    ["MOTA ↑",       "55.19", "47.19", delta_p("−8.00", False), "uav0000117 dominates"],
    ["IDF1 ↑",       "73.57", "70.86", delta_p("−2.71", False), ""],
    ["Recall ↑",     "77.70%","79.44%",delta_p("+1.74%", True), "✅ More GT vehicles found"],
    ["Precision ↑",  "77.60%","71.20%",delta_p("−6.40%",False), "⚠ UAV sequence precision hit"],
    ["FP ↓",     "160,558","230,096",  delta_p("+69,538",False), "58,115 from uav0000117"],
    ["FN ↓",     "159,710","147,213",  delta_p("−12,497",True),  "✅ C2 finds 12k more GT"],
    ["IDSW ↓",       "634",   "819",   delta_p("+185",   False), "More tracks → more switches"],
    ["MT ↑",      "69.73%", "73.06%",  delta_p("+3.33%", True),  "✅ More sequences tracked"],
    ["ML ↓",       "8.52%",  "7.51%",  delta_p("−1.01%", True),  "✅ Fewer sequences lost"],
    ["LocA ↑",     "88.75",  "88.67",  delta_p("−0.08"),         "✅ Stable"],
]

col_w = [2.8*cm, 2.4*cm, 2.4*cm, 2.4*cm, CONTENT_WIDTH-10.0*cm]
rt = Table(results_data, colWidths=col_w, repeatRows=1)
rt.setStyle(TableStyle([
    ("BACKGROUND",    (0,0),(-1,0),  C_LIGHT),
    ("LINEBELOW",     (0,0),(-1,0),  1.5, C_BORDER),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [white, HexColor("#f8fafc")]),
    ("LINEBELOW",     (0,1),(-1,-1), 0.5, C_BORDER),
    ("BOX",           (0,0),(-1,-1), 1,   C_BORDER),
    ("TOPPADDING",    (0,0),(-1,-1), 6),
    ("BOTTOMPADDING", (0,0),(-1,-1), 6),
    ("LEFTPADDING",   (0,0),(-1,-1), 8),
    ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0,0),(-1,0),  8),
    ("TEXTCOLOR",     (0,0),(-1,0),  C_MUTED),
    ("BACKGROUND",    (0,1),(- 1,1), C_BG_BLUE),  # HOTA row
    ("FONTNAME",      (0,1),(-1,1),  "Helvetica-Bold"),
    ("BACKGROUND",    (0,4),(-1,4),  C_BG_BLUE),  # MOTA row
    ("FONTNAME",      (0,4),(-1,4),  "Helvetica-Bold"),
    ("BACKGROUND",    (0,5),(-1,5),  C_BG_BLUE),  # IDF1 row
    ("FONTNAME",      (0,5),(-1,5),  "Helvetica-Bold"),
]))
story += [rt, SP(0.3)]

# ── 9. Analysis ───────────────────────────────────────────────────────────────
story += section_heading(9, "Analysis & Interpretation")

story.append(P("<b>What C2 Got Right</b>", "h3"))
story += [
    B("Recall +1.74%, FN −12,497 — C2 correctly lowers τ for dark frames, recovering genuine vehicles"),
    B("MT +3.33%, ML −1.01% — More sequences mostly tracked; fewer mostly lost"),
    B("AssA −0.10 (essentially zero) — confirms C2 only affects detection admission, not track memory"),
    B("LocA −0.08 (negligible) — bounding box quality unchanged"),
    B("Clamp verified: all tau values in [0.3, 0.7] across 30 probed frames"),
]

story.append(SP(0.15))
story.append(P("<b>The uav0000117 Problem</b>", "h3"))
story.append(P(
    "This single UAV night sequence contributed 58,115 of the 69,538 total FP increase (84%). "
    "Mean brightness 0.1805 → τ correctly dropped to ~0.341 (near τ_min=0.3). However, at UAV "
    "altitude/resolution, MeMOTR's 300 det-queries consistently score above 0.34 even for "
    "non-vehicles — domain shift, not a calibration error. Fix: include UAV-altitude sequences "
    "in C2 adapter training (Phase 6 proper with IOTD data)."))
story.append(P(
    "<b>DETRAC-Only C2 Performance:</b> Excluding uav0000117, FP increase drops to only +11,423 "
    "with all FN/MT gains intact. Multiple DETRAC sequences show small HOTA/MOTA improvements. "
    "This validates the C2 mechanism works correctly on its training domain."))

# ── 10. Roadmap ───────────────────────────────────────────────────────────────
story += section_heading(10, "Roadmap — Remaining Work")
story += data_table(
    ["Phase", "Task", "Blocker", "Priority"],
    [
        ["4",  "IOTD annotation export (≥3 sequences)", "Manual annotation", "Critical"],
        ["4",  "Perspective calibration for C1",         "Phase 4 annotation","Critical"],
        ["6",  "Retrain C2 with IOTD night/dusk data",   "Phase 4",           "Critical"],
        ["6",  "Train C1 PositionMLP",                   "Phase 4",           "High"],
        ["6",  "Joint C1+C2 evaluation on IOTD test",    "C1+C2 training",    "High"],
        ["6",  "Ablation table (BL vs C2 vs C1 vs C1+C2)","All runs",         "High"],
        ["7",  "Final paper results (Section 12)",        "All eval runs",     "Normal"],
    ],
    col_widths=[1.5*cm, 6.5*cm, 4.0*cm, CONTENT_WIDTH-12.0*cm]
)

story.append(SP(0.2))
story += callout(
    "Phase 5/6 interim baselines complete · Raw TrackEval summaries preserved · "
    "Clamp verification done · Failure mode diagnosed · teacher_summary.md populated · "
    "All checkpoints saved (baseline epochs 0–4, C2 adapter epochs 0–4)",
    title="✅ Interim Deliverables Completed", bg=C_BG_GREEN, border=C_SUCCESS)

# ── Build ──────────────────────────────────────────────────────────────────────
print(f"Building PDF: {OUTPUT_PATH}")
doc.build(story)
print(f"Done -- {os.path.getsize(OUTPUT_PATH):,} bytes -> {OUTPUT_PATH}")
