"""Generate the CONDENSED weekly-update slideshow (4 slides, technical standup) for open-code-bench.

Re-run after edits:  .venv\\Scripts\\python.exe scripts\\build_weekly_deck_short.py
Output: open-code-bench-weekly-update-2026-06-26-summary.pptx (repo root).
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

NAVY = RGBColor(0x0F, 0x2A, 0x43)
ACCENT = RGBColor(0x16, 0x9B, 0xD7)
DARK = RGBColor(0x23, 0x29, 0x30)
GRAY = RGBColor(0x8A, 0x93, 0x9B)
LIGHT = RGBColor(0xEE, 0xF3, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SUBDARK = RGBColor(0x4A, 0x52, 0x5A)
FOOTER = "© 2026 BrainChip Holdings Ltd. All rights reserved. Internal use only"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def _footer(slide, idx):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(7.04), Inches(10.5), Inches(0.36))
    p = tb.text_frame.paragraphs[0]
    p.text = FOOTER; p.font.size = Pt(9); p.font.color.rgb = GRAY
    nb = slide.shapes.add_textbox(Inches(12.3), Inches(7.04), Inches(0.8), Inches(0.36))
    np = nb.text_frame.paragraphs[0]
    np.text = str(idx); np.alignment = PP_ALIGN.RIGHT; np.font.size = Pt(9); np.font.color.rgb = GRAY


def _header(slide, title, kicker):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(12.3), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    p.text = title; p.font.size = Pt(32); p.font.bold = True; p.font.color.rgb = NAVY
    if kicker:
        kb = slide.shapes.add_textbox(Inches(0.52), Inches(1.2), Inches(12.3), Inches(0.4))
        kp = kb.text_frame.paragraphs[0]
        kp.text = kicker; kp.font.size = Pt(14); kp.font.italic = True; kp.font.color.rgb = ACCENT
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.62), Inches(2.4), Pt(3))
    line.fill.solid(); line.fill.fore_color.rgb = ACCENT; line.line.fill.background()


def bullets_slide(idx, title, kicker, bullets):
    slide = prs.slides.add_slide(BLANK)
    _header(slide, title, kicker)
    body = slide.shapes.add_textbox(Inches(0.6), Inches(2.0), Inches(12.1), Inches(4.8))
    bf = body.text_frame; bf.word_wrap = True
    for i, (text, level) in enumerate(bullets):
        p = bf.paragraphs[0] if i == 0 else bf.add_paragraph()
        p.text = ("•  " if level == 0 else "–  ") + text
        p.level = level
        p.font.size = Pt(20 if level == 0 else 16)
        p.font.color.rgb = DARK if level == 0 else SUBDARK
        p.space_after = Pt(12 if level == 0 else 5)
    _footer(slide, idx)


def results_slide(idx):
    slide = prs.slides.add_slide(BLANK)
    _header(slide, "Results & Key Findings", "Two benchmarks × four self-hosted backends · pass@1, greedy")
    headers = ["Model", "Backend", "HumanEval+", "BCB-hard"]
    rows = [
        ["qwen2.5-coder 1.5B", "Raspberry Pi 5 · Ollama", "0.610", "—"],
        ["qwen2.5-coder 7B", "DGX Spark · vLLM (GB10)", "0.823", "0.224"],
        ["qwen2.5-coder 32B", "DGX Spark · vLLM (GB10)", "0.866", "0.385"],
        ["Qwen2.5-72B-Instruct (AWQ)", "DGX Spark · vLLM (GB10)", "0.805", "0.324"],
    ]
    hi = 2
    nrows = len(rows) + 1
    gt = slide.shapes.add_table(nrows, 4, Inches(0.6), Inches(1.95),
                                Inches(12.1), Inches(0.48 * nrows)).table
    for c, w in enumerate([4.4, 4.3, 1.8, 1.6]):
        gt.columns[c].width = Inches(w)
    for c, h in enumerate(headers):
        cell = gt.cell(0, c); cell.text = h
        pa = cell.text_frame.paragraphs[0]
        pa.font.size = Pt(15); pa.font.bold = True; pa.font.color.rgb = WHITE
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY; cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = gt.cell(r, c); cell.text = str(val)
            pa = cell.text_frame.paragraphs[0]
            bold = (r - 1) == hi
            pa.font.size = Pt(14); pa.font.bold = bold
            pa.font.color.rgb = NAVY if bold else DARK
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xDB, 0xEE, 0xF8) if bold else (LIGHT if r % 2 else WHITE)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    # findings under the table
    fb = slide.shapes.add_textbox(Inches(0.6), Inches(4.55), Inches(12.1), Inches(2.3))
    ff = fb.text_frame; ff.word_wrap = True
    findings = [
        ("BCB-hard discriminates far better: 7B→32B is +72% (0.224→0.385) vs +5% on HumanEval+", 0),
        ("HumanEval+ is near-saturated for capable coders; BCB-hard has real headroom", 1),
        ("Same ranking on both: 32B-Coder > 72B-Instruct-AWQ > 7B-Coder", 0),
        ("Code-specialization + full precision beats raw size + 4-bit quantization", 1),
    ]
    for i, (t, lvl) in enumerate(findings):
        p = ff.paragraphs[0] if i == 0 else ff.add_paragraph()
        p.text = ("►  " if lvl == 0 else "–  ") + t
        p.level = lvl
        p.font.size = Pt(16 if lvl == 0 else 13)
        p.font.bold = lvl == 0
        p.font.color.rgb = NAVY if lvl == 0 else SUBDARK
        p.space_after = Pt(4)
    _footer(slide, idx)


# ---- slide 1: overview (navy title slide doubling as 'at a glance') ----
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
t = s.shapes.add_textbox(Inches(0.9), Inches(0.7), Inches(11.5), Inches(1.0))
tp = t.text_frame.paragraphs[0]
tp.text = "open-code-bench — This Week at a Glance"
tp.font.size = Pt(36); tp.font.bold = True; tp.font.color.rgb = WHITE
st = s.shapes.add_textbox(Inches(0.92), Inches(1.6), Inches(11.5), Inches(0.5))
stp = st.text_frame.paragraphs[0]
stp.text = "Week of June 23–26, 2026"; stp.font.size = Pt(16); stp.font.color.rgb = ACCENT
bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.92), Inches(2.15), Inches(2.6), Pt(3))
bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
body = s.shapes.add_textbox(Inches(0.92), Inches(2.5), Inches(11.6), Inches(4.0))
bf = body.text_frame; bf.word_wrap = True
glance = [
    "From a single-benchmark MVP to a 2-benchmark, multi-backend framework — in one week",
    "Stood up vLLM on the DGX Spark (GB10): Qwen-Coder 7B/32B (FP16) + 72B (AWQ 4-bit)",
    "Refactored into a pluggable benchmark framework; added BigCodeBench as the 2nd benchmark",
    "7 scored runs across 2 benchmarks + 4 backends; results + findings published to the README",
]
for i, txt in enumerate(glance):
    p = bf.paragraphs[0] if i == 0 else bf.add_paragraph()
    p.text = "•  " + txt; p.font.size = Pt(20)
    p.font.color.rgb = RGBColor(0xE6, 0xEF, 0xF6); p.space_after = Pt(16)
ftb = s.shapes.add_textbox(Inches(0.9), Inches(7.0), Inches(11.5), Inches(0.4))
fp = ftb.text_frame.paragraphs[0]
fp.text = FOOTER; fp.font.size = Pt(9); fp.font.color.rgb = RGBColor(0x9F, 0xB3, 0xC4)

# ---- slide 2: what we built ----
bullets_slide(2, "What We Built", "The platform under the leaderboard", [
    ("One gateway, decoupled pipeline:", 0),
    ("generate (gateway → backends) → score (hardened sandbox) → results + full provenance", 1),
    ("vLLM on GB10: official ARM64 image; FP16 + AWQ 4-bit both work; ~105 GB-weight ceiling mapped", 0),
    ("Pluggable framework: a new benchmark = one plugin (prompts + native evaluator) + a YAML spec", 0),
    ("HumanEval+ and BigCodeBench are the two worked examples of the same interface", 1),
    ("Security: untrusted model code runs only in a network-isolated, cap-dropped, resource-capped container", 0),
])

# ---- slide 3: results & findings ----
results_slide(3)

# ---- slide 4: next steps ----
bullets_slide(4, "Next Steps", "Roadmap from here", [
    ("Broaden BigCodeBench coverage — more models; the full 1140-task subset", 0),
    ("Optimize: persist the BCB ground-truth cache (~245s recomputed each run today)", 0),
    ("Postgres results store (D7) when cross-run comparison gets tedious", 0),
    ("Next benchmarks: LiveCodeBench (stdin/stdout), then Aider (multi-turn / workspace)", 0),
])

out = Path(__file__).resolve().parent.parent / "open-code-bench-weekly-update-2026-06-26-summary.pptx"
prs.save(str(out))
print(f"wrote {out}  ({len(prs.slides._sldIdLst)} slides)")
