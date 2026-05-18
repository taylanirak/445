"""Render Group24_final_report.md to a PDF via reportlab.

Mirrors the milestone PDF builder (same lightweight Markdown subset, no pandoc
/ LaTeX dependency) and additionally substitutes the live ``results.json``
numbers through ``src.report_fill`` so the PDF always matches the run.  The
final report limit is 8 pages excluding references/appendix; the page count is
printed so the team can verify the limit.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)
from reportlab.lib.units import inch

FINAL = Path(__file__).resolve().parent
sys.path.insert(0, str(FINAL))
from src.report_fill import load_filled  # noqa: E402

MD = FINAL / "Group24_final_report.md"
PDF = FINAL / "Group24_final_report.pdf"

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica",
                      fontSize=9.5, leading=11.8, spaceAfter=3, alignment=4)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                    fontSize=13, leading=16, spaceBefore=6, spaceAfter=3)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                    fontSize=11, leading=13, spaceBefore=5, spaceAfter=2)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4,
                        spaceAfter=1)


def md_inline(text: str) -> str:
    text = text.replace("&", "&amp;")
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="9">\1</font>', text)
    return text


def parse_table(lines, i):
    header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    i += 2
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
        i += 1
    return header, rows, i


def render():
    raw = load_filled(MD)
    if raw.lstrip().startswith("---"):
        s = raw.lstrip()
        end = s.find("---", 3)
        yaml, body_text = s[3:end], s[end + 3:]
    else:
        yaml, body_text = "", raw

    def y(field):
        m = re.search(rf'{field}:\s*"?([^"\n]+)"?', yaml)
        return m.group(1) if m else None

    doc = SimpleDocTemplate(str(PDF), pagesize=LETTER,
                            leftMargin=0.7 * inch, rightMargin=0.7 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.55 * inch,
                            title="SemEval-2026 Task 5 Final Report")
    flow = []
    if y("title"):
        flow.append(Paragraph(f"<b>{y('title')}</b>",
                    ParagraphStyle("t", fontName="Helvetica-Bold",
                                   fontSize=14, alignment=1, spaceAfter=2)))
    if y("subtitle"):
        flow.append(Paragraph(f"<i>{y('subtitle')}</i>",
                    ParagraphStyle("st", fontName="Helvetica-Oblique",
                                   fontSize=10, alignment=1, spaceAfter=4)))
    if y("author"):
        flow.append(Paragraph(y("author"),
                    ParagraphStyle("a", fontName="Helvetica", fontSize=10,
                                   alignment=1, spaceAfter=1)))
    if y("date"):
        flow.append(Paragraph(y("date"),
                    ParagraphStyle("d", fontName="Helvetica", fontSize=9,
                                   alignment=1, spaceAfter=8)))

    lines = body_text.splitlines()
    i = 0
    buf: list[str] = []

    def flush():
        if buf:
            flow.append(Paragraph(md_inline(" ".join(buf)), body))
            buf.clear()

    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("# "):
            flush(); flow.append(Paragraph(md_inline(stripped[2:]), h1)); i += 1
        elif stripped.startswith("## "):
            flush(); flow.append(Paragraph(md_inline(stripped[3:]), h2)); i += 1
        elif stripped.startswith("### "):
            flush()
            flow.append(Paragraph(f"<b>{md_inline(stripped[4:])}</b>", h2))
            i += 1
        elif (stripped.startswith("|") and i + 1 < len(lines)
              and set(lines[i + 1].strip()).issubset(set("|-: "))):
            flush()
            header, rows, i = parse_table(lines, i)
            data = [[Paragraph(md_inline(c), body) for c in header]]
            for r in rows:
                while len(r) < len(header):
                    r.append("")
                data.append([Paragraph(md_inline(c), body)
                             for c in r[:len(header)]])
            t = Table(data, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]))
            flow.append(KeepTogether([t, Spacer(1, 4)]))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            flush()
            flow.append(Paragraph(md_inline(stripped[2:]), bullet,
                                  bulletText="•"))
            i += 1
        elif re.match(r"^\d+\.\s", stripped):
            flush()
            num, rest = stripped.split(".", 1)
            flow.append(Paragraph(md_inline(rest.strip()), bullet,
                                  bulletText=f"{num}."))
            i += 1
        elif stripped.startswith("```"):
            flush()
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i]); i += 1
            i += 1
            cs = ParagraphStyle("code", parent=body, fontName="Courier",
                                fontSize=8.5, leading=10.5, leftIndent=12,
                                backColor=colors.whitesmoke, spaceAfter=4)
            flow.append(Paragraph("<br/>".join(
                ln.replace("&", "&amp;").replace(" ", "&nbsp;")
                for ln in code_lines), cs))
        elif stripped == "---":
            flush(); flow.append(Spacer(1, 4)); i += 1
        elif stripped == "":
            flush(); i += 1
        else:
            buf.append(stripped); i += 1
    flush()

    page_count = {"n": 0}

    def _count(canvas, _doc):
        page_count["n"] = _doc.page

    doc.build(flow, onLaterPages=_count, onFirstPage=_count)
    print(f"Wrote {PDF}  ({page_count['n']} pages; final-report limit is 8 "
          f"excluding references/appendix)")


if __name__ == "__main__":
    render()
