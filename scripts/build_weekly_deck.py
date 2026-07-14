"""Generate the weekly-update slideshow (technical audience) for open-code-bench.

Renders a 16:9 .pptx from the structured content below. Re-run to regenerate after edits:
    .venv\\Scripts\\python.exe scripts\\build_weekly_deck.py
Output: open-code-bench-weekly-update-2026-06-26.pptx (repo root).
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

# ---- palette ----
NAVY = RGBColor(0x0F, 0x2A, 0x43)
ACCENT = RGBColor(0x16, 0x9B, 0xD7)
DARK = RGBColor(0x23, 0x29, 0x30)
GRAY = RGBColor(0x8A, 0x93, 0x9B)
LIGHT = RGBColor(0xEE, 0xF3, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FOOTER = "© 2026 BrainChip Holdings Ltd. All rights reserved. Internal use only"

EMU_W, EMU_H = Inches(13.333), Inches(7.5)
prs = Presentation()
prs.slide_width = EMU_W
prs.slide_height = EMU_H
BLANK = prs.slide_layouts[6]


def _footer(slide, idx):
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(7.04), Inches(10.5), Inches(0.36))
    p = tb.text_frame.paragraphs[0]
    p.text = FOOTER
    p.font.size = Pt(9)
    p.font.color.rgb = GRAY
    nb = slide.shapes.add_textbox(Inches(12.3), Inches(7.04), Inches(0.8), Inches(0.36))
    np = nb.text_frame.paragraphs[0]
    np.text = str(idx)
    np.alignment = PP_ALIGN.RIGHT
    np.font.size = Pt(9)
    np.font.color.rgb = GRAY


def content_slide(idx, title, kicker, bullets):
    slide = prs.slides.add_slide(BLANK)
    # title
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(12.3), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = NAVY
    if kicker:
        kb = slide.shapes.add_textbox(Inches(0.52), Inches(1.12), Inches(12.3), Inches(0.4))
        kp = kb.text_frame.paragraphs[0]
        kp.text = kicker
        kp.font.size = Pt(13)
        kp.font.italic = True
        kp.font.color.rgb = ACCENT
    # accent rule
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.55), Inches(2.2), Pt(3))
    line.fill.solid(); line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()
    # bullets
    body = slide.shapes.add_textbox(Inches(0.6), Inches(1.85), Inches(12.1), Inches(5.0))
    bf = body.text_frame
    bf.word_wrap = True
    for i, (text, level) in enumerate(bullets):
        p = bf.paragraphs[0] if i == 0 else bf.add_paragraph()
        prefix = "•  " if level == 0 else "–  "
        p.text = prefix + text
        p.level = level
        p.font.size = Pt(18 if level == 0 else 15)
        p.font.color.rgb = DARK if level == 0 else GRAY
        p.space_after = Pt(7 if level == 0 else 3)
        if level == 1:
            p.font.color.rgb = RGBColor(0x4A, 0x52, 0x5A)
    _footer(slide, idx)
    return slide


def table_slide(idx, title, kicker, headers, rows, note, highlight_row=None, col_widths=None):
    slide = prs.slides.add_slide(BLANK)
    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(12.3), Inches(0.9))
    p = tb.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(30); p.font.bold = True; p.font.color.rgb = NAVY
    if kicker:
        kb = slide.shapes.add_textbox(Inches(0.52), Inches(1.12), Inches(12.3), Inches(0.4))
        kp = kb.text_frame.paragraphs[0]
        kp.text = kicker; kp.font.size = Pt(13); kp.font.italic = True; kp.font.color.rgb = ACCENT
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(1.55), Inches(2.2), Pt(3))
    line.fill.solid(); line.fill.fore_color.rgb = ACCENT; line.line.fill.background()

    nrows, ncols = len(rows) + 1, len(headers)
    gt = slide.shapes.add_table(nrows, ncols, Inches(0.6), Inches(1.95),
                                Inches(12.1), Inches(0.5 * nrows)).table
    if col_widths:
        for c, w in enumerate(col_widths):
            gt.columns[c].width = Inches(w)
    # header
    for c, h in enumerate(headers):
        cell = gt.cell(0, c)
        cell.text = h
        para = cell.text_frame.paragraphs[0]
        para.font.size = Pt(14); para.font.bold = True; para.font.color.rgb = WHITE
        cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    # body
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = gt.cell(r, c)
            cell.text = str(val)
            para = cell.text_frame.paragraphs[0]
            para.font.size = Pt(13)
            bold = highlight_row is not None and (r - 1) == highlight_row
            para.font.bold = bold
            para.font.color.rgb = NAVY if bold else DARK
            cell.fill.solid()
            if bold:
                cell.fill.fore_color.rgb = RGBColor(0xDB, 0xEE, 0xF8)
            else:
                cell.fill.fore_color.rgb = LIGHT if r % 2 else WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if note:
        nb = slide.shapes.add_textbox(Inches(0.6), Inches(2.15 + 0.5 * nrows), Inches(12.1), Inches(1.6))
        nf = nb.text_frame; nf.word_wrap = True
        for i, ln in enumerate(note):
            np = nf.paragraphs[0] if i == 0 else nf.add_paragraph()
            np.text = ln
            np.font.size = Pt(12); np.font.color.rgb = GRAY; np.space_after = Pt(3)
    _footer(slide, idx)
    return slide


# ---- title slide ----
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, EMU_W, EMU_H)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(3.55), Inches(3.0), Pt(4))
bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
t = s.shapes.add_textbox(Inches(0.9), Inches(2.3), Inches(11.5), Inches(1.2))
tp = t.text_frame.paragraphs[0]
tp.text = "open-code-bench — Weekly Update"
tp.font.size = Pt(44); tp.font.bold = True; tp.font.color.rgb = WHITE
sub = s.shapes.add_textbox(Inches(0.92), Inches(3.75), Inches(11.5), Inches(1.4))
sf = sub.text_frame; sf.word_wrap = True
for i, (txt, sz, col) in enumerate([
    ("Multi-backend LLM coding-benchmark platform", 22, RGBColor(0xCF, 0xE3, 0xF2)),
    ("Week of June 23–26, 2026", 16, ACCENT),
]):
    p = sf.paragraphs[0] if i == 0 else sf.add_paragraph()
    p.text = txt; p.font.size = Pt(sz); p.font.color.rgb = col; p.space_after = Pt(6)
fb = s.shapes.add_textbox(Inches(0.9), Inches(7.0), Inches(11.5), Inches(0.4))
fp = fb.text_frame.paragraphs[0]
fp.text = FOOTER; fp.font.size = Pt(9); fp.font.color.rgb = RGBColor(0x9F, 0xB3, 0xC4)

# ---- content slides ----
content_slide(2, "This week at a glance", "From a HumanEval+ MVP to a two-benchmark framework", [
    ("Stood up end-to-end HumanEval+ scoring → first real pass@1 numbers", 0),
    ("Multi-backend leaderboard: Raspberry Pi (edge CPU) + DGX Spark (GB10) via vLLM", 0),
    ("vLLM bring-up on GB10 — FP16 (7B/32B) and AWQ 4-bit (72B) both working", 0),
    ("Refactored two bespoke scripts → a pluggable Benchmark framework (src/ocb)", 0),
    ("Added BigCodeBench as the 2nd benchmark — the framework seam is proven", 0),
    ("7 results across 2 benchmarks + 2 clean findings; README leaderboard published", 0),
])

content_slide(3, "Architecture — one gateway, decoupled stages", "Generate where the models live; score where the sandbox lives", [
    ("A single OpenAI-compatible gateway (LiteLLM proxy) fronts every backend", 0),
    ("run → gateway (generate, tagged by run_id) → sandbox (run tests, score) → results + provenance", 0),
    ("Generate and score are decoupled steps — they can run on different hosts / networks", 0),
    ("Generation is owned in-house for uniform cost/latency/provenance; scoring is delegated to each", 0),
    ("benchmark's native evaluator (EvalPlus, BigCodeBench)", 1),
    ("Every call tagged: run_id, model, sampling params, dataset version + hash, git commit", 0),
])

content_slide(4, "Backends behind the gateway", "Heterogeneous endpoints, one logical API", [
    ("Raspberry Pi 5 · Ollama — qwen2.5-coder 1.5B (Q4_K_M): the slow CPU edge data point", 0),
    ("DGX Spark (GB10) · vLLM — qwen2.5-coder 7B / 32B (FP16), Qwen2.5-72B-Instruct (AWQ 4-bit)", 0),
    ("Amazon Bedrock + Anthropic Claude API — planned (cloud keys live only on the gateway)", 0),
    ("SSH is control-plane only (start server / load model); inference is HTTP over the LAN", 0),
    ("Host IPs externalized to a gitignored .env — no addresses in the repo", 0),
])

content_slide(5, "vLLM on the DGX Spark (GB10)", "ARM64 + sm_121 + CUDA 13 — no custom image needed", [
    ("Official vllm/vllm-openai image (has a linux/arm64 manifest) runs on GB10 out of the box", 0),
    ("--gpus all works via the nvidia-container-toolkit hook — no nvidia runtime registered", 0),
    ("Both FP16 (7B/32B) and AWQ 4-bit (72B) kernels work on sm_121", 0),
    ("Unified-memory ceiling found: models with <~105 GB of weights fit", 0),
    ("235B-4bit (124 GB) does NOT fit — caught via the HF API before a wasted download", 1),
    ("Request concurrency (32) lets vLLM batch in-flight requests → 148–164 tasks in ~3–6 min", 0),
])

table_slide(6, "Results — HumanEval+ (164 tasks)", "temperature 0, single sample (pass@1), self-hosted",
    ["Model", "Backend", "pass@1", "base pass@1"],
    [
        ["qwen2.5-coder 1.5B (Q4_K_M)", "Raspberry Pi 5 · Ollama", "0.610", "0.665"],
        ["qwen2.5-coder 7B", "DGX Spark · vLLM (GB10)", "0.823", "0.872"],
        ["qwen2.5-coder 32B", "DGX Spark · vLLM (GB10)", "0.866", "0.909"],
        ["Qwen2.5-72B-Instruct (AWQ 4-bit)", "DGX Spark · vLLM (GB10)", "0.805", "0.848"],
    ],
    ["pass@1 = HumanEval+ (base + extra tests); base pass@1 = original HumanEval tests only.",
     "Dataset HumanEvalPlus v0.1.10 (fe585eb4…), EvalPlus 0.3.1. All runs 164/164 complete.",
     "The 72B is a general Instruct model + 4-bit — which is why it trails the 32B-Coder."],
    highlight_row=2, col_widths=[4.2, 4.0, 2.0, 1.9])

content_slide(7, "Framework refactor (Phases A + B)", "Bespoke scripts → a reusable package", [
    ("Before: two one-off scripts (generate, score); adding a benchmark meant copy-paste", 0),
    ("After: src/ocb package with clean seams —", 0),
    ("GatewayClient · Benchmark ABC + plugin registry · SandboxRunner · spec-driven runner · provenance", 1),
    ("Run specs (YAML) replace ad-hoc flags: benchmark + models + sampling + concurrency", 0),
    ("Validated behavior-identical: re-scored the 32B run → 0.866 unchanged", 0),
    ("Old scripts kept as thin wrappers over the runner — nothing breaks for existing users", 0),
])

content_slide(8, "The benchmark plugin seam", "Adding a benchmark = one subclass + a spec", [
    ("A plugin owns: build_prompt, extract_solution, evaluate() → its native evaluator", 0),
    ("The runner owns: the run() loop (k-sampling), scoring orchestration, provenance, status tracking", 0),
    ("infra-error ≠ wrong answer; truncated samples excluded from pass@1, never scored as wrong", 1),
    ("Per-benchmark knobs, no core changes: sandbox_image, sandbox_read_only, sandbox_auto_confirm", 0),
    ("HumanEval+ and BigCodeBench are now the two worked examples of the same interface", 0),
])

content_slide(9, "BigCodeBench integration (Phase C)", "The second plugin that proves the seam", [
    ("Same generate/score split as HumanEval+, but different in every detail it touches", 0),
    ("Two axes: split (complete | instruct) × subset (full 1140 | hard 148)", 0),
    ("Built an offline hardened sandbox image: dataset + 139 task libraries + ground-truth baked in,", 0),
    ("HF offline → the evaluator runs fully air-gapped under --network=none", 1),
    ("Solved live: writable rootfs (GT pickle cache) + stdin auto-confirm ([Y/N] prompt EOFs over SSH)", 0),
    ("Validated end-to-end: 7B → pass@1 0.224, matching the official scorer within rounding", 0),
])

content_slide(10, "Sandbox hardening (D11) — the trust boundary", "Untrusted model code only ever runs in the container", [
    ("Hardened docker run, flags at the call site (auditable), not baked into the image:", 0),
    ("--network=none · --cap-drop=ALL · --security-opt=no-new-privileges", 1),
    ("--read-only rootfs (strict profile; relaxed only where an evaluator must write)", 1),
    ("--tmpfs scratch · --pids-limit · --cpus · --memory · non-root user", 1),
    ("Remote path: scp samples in → run → scp results out → wipe the remote work dir", 0),
    ("Datasets + ground truth pre-baked at image build → no network needed at eval time", 0),
])

table_slide(11, "Results — BigCodeBench-hard (148 tasks, complete)", "The harder benchmark, scored this week",
    ["Model", "Backend", "pass@1", "Complete"],
    [
        ["qwen2.5-coder 7B", "DGX Spark · vLLM (GB10)", "0.224", "147/148"],
        ["qwen2.5-coder 32B", "DGX Spark · vLLM (GB10)", "0.385", "148/148"],
        ["Qwen2.5-72B-Instruct (AWQ 4-bit)", "DGX Spark · vLLM (GB10)", "0.324", "148/148"],
    ],
    ["pass@1 over fairly-attempted (matches official scorer within rounding).",
     "Dataset BigCodeBench v0.1.4 hard subset (f8d6f960…), bigcodebench 0.2.5.",
     "Pi 1.5B skipped: a 1.5B on the hardest subset is ~floor and a multi-hour CPU run."],
    highlight_row=1, col_widths=[4.2, 4.0, 2.0, 1.9])

content_slide(12, "Key findings", "What two benchmarks across four backends tell us", [
    ("BigCodeBench-hard discriminates far better than HumanEval+", 0),
    ("7B→32B jump: +72% on BCB-hard (0.224→0.385) vs +5% on HumanEval+ (0.823→0.866)", 1),
    ("HumanEval+ is near-saturated for capable coders; BCB-hard has real headroom", 1),
    ("Same ranking on both benchmarks: 32B-Coder > 72B-Instruct-AWQ > 7B-Coder", 0),
    ("Code-specialization + full precision beats raw size + 4-bit quantization for coding", 1),
    ("Takeaway: BCB-hard is the more useful ranking benchmark for capable coding models", 0),
])

content_slide(13, "Next steps", "Roadmap from here", [
    ("Done: README leaderboard + quickstart refreshed to the new framework", 0),
    ("Breadth: more models on BCB-hard; consider the full 1140-task subset", 0),
    ("Optimization: persist the BigCodeBench ground-truth cache (~245s recomputed each run today)", 0),
    ("Results store: Postgres system-of-record (D7) when cross-run comparison gets tedious", 0),
    ("More benchmarks: LiveCodeBench (stdin/stdout), then Aider (multi-turn / workspace profile)", 0),
])

out = Path(__file__).resolve().parent.parent / "open-code-bench-weekly-update-2026-06-26.pptx"
prs.save(str(out))
print(f"wrote {out}  ({len(prs.slides._sldIdLst)} slides)")
