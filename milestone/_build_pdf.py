"""Render GroupXX_milestone_report.md to a four-page PDF via reportlab.

Kept deliberately simple — the master is the .md file; this produces the PDF
for submission. Re-run after editing the .md.
"""
from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle, KeepTogether)

REPO = Path(__file__).resolve().parent.parent
MD   = REPO / "milestone" / "GroupXX_milestone_report.md"
PDF  = REPO / "milestone" / "GroupXX_milestone_report.pdf"

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica",
                     fontSize=9.5, leading=11.8, spaceAfter=3, alignment=4)  # justified
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                   fontSize=13, leading=16, spaceBefore=6, spaceAfter=3)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                   fontSize=11, leading=13, spaceBefore=5, spaceAfter=2)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=1)


def md_inline(text: str) -> str:
    # bold **...** and italics *...*, and `code`
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="9">\1</font>', text)
    # escape stray ampersands
    text = text.replace("&", "&amp;")
    # undo the escapes we made inside tags
    text = text.replace("<b>", "<b>").replace("<i>", "<i>")
    return text


def parse_table(lines, i):
    # Markdown pipe table. Expect header, separator, rows.
    header = [c.strip() for c in lines[i].strip().strip("|").split("|")]
    i += 2  # skip separator
    rows = []
    while i < len(lines) and lines[i].strip().startswith("|"):
        rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
        i += 1
    return header, rows, i


def render():
    raw = MD.read_text(encoding="utf-8")
    # Strip YAML front matter.
    if raw.startswith("---"):
        end = raw.find("---", 3)
        yaml = raw[3:end]
        body_text = raw[end+3:]
    else:
        yaml, body_text = "", raw

    title = re.search(r'title:\s*"?([^"\n]+)"?', yaml)
    subtitle = re.search(r'subtitle:\s*"?([^"\n]+)"?', yaml)
    author = re.search(r'author:\s*"?([^"\n]+)"?', yaml)
    date = re.search(r'date:\s*"?([^"\n]+)"?', yaml)

    doc = SimpleDocTemplate(str(PDF), pagesize=LETTER,
                            leftMargin=0.7*inch, rightMargin=0.7*inch,
                            topMargin=0.6*inch, bottomMargin=0.55*inch,
                            title="SemEval-2026 Task 5 Milestone")
    flow = []

    if title:
        flow.append(Paragraph(f"<b>{title.group(1)}</b>",
                              ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=14, alignment=1, spaceAfter=2)))
    if subtitle:
        flow.append(Paragraph(f"<i>{subtitle.group(1)}</i>",
                              ParagraphStyle("st", fontName="Helvetica-Oblique", fontSize=10, alignment=1, spaceAfter=4)))
    if author:
        flow.append(Paragraph(author.group(1),
                              ParagraphStyle("a", fontName="Helvetica", fontSize=10, alignment=1, spaceAfter=1)))
    if date:
        flow.append(Paragraph(date.group(1),
                              ParagraphStyle("d", fontName="Helvetica", fontSize=9, alignment=1, spaceAfter=8)))

    lines = body_text.splitlines()
    i = 0
    buf: list[str] = []

    def flush():
        if buf:
            flow.append(Paragraph(md_inline(" ".join(buf)), body))
            buf.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("# "):
            flush()
            flow.append(Paragraph(md_inline(stripped[2:]), h1))
            i += 1
        elif stripped.startswith("## "):
            flush()
            flow.append(Paragraph(md_inline(stripped[3:]), h2))
            i += 1
        elif stripped.startswith("### "):
            flush()
            flow.append(Paragraph(f"<b>{md_inline(stripped[4:])}</b>", h2))
            i += 1
        elif stripped.startswith("|") and i + 1 < len(lines) and set(lines[i+1].strip()).issubset(set("|-: ")):
            flush()
            header, rows, i = parse_table(lines, i)
            data = [[Paragraph(md_inline(c), body) for c in header]]
            for r in rows:
                # pad/truncate to header length
                while len(r) < len(header): r.append("")
                data.append([Paragraph(md_inline(c), body) for c in r[:len(header)]])
            t = Table(data, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("LEFTPADDING", (0,0), (-1,-1), 4),
                ("RIGHTPADDING", (0,0), (-1,-1), 4),
                ("TOPPADDING", (0,0), (-1,-1), 2),
                ("BOTTOMPADDING", (0,0), (-1,-1), 2),
            ]))
            flow.append(KeepTogether([t, Spacer(1, 4)]))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            flush()
            flow.append(Paragraph(md_inline(stripped[2:]), bullet, bulletText="•"))
            i += 1
        elif re.match(r"^\d+\.\s", stripped):
            flush()
            num, rest = stripped.split(".", 1)
            flow.append(Paragraph(md_inline(rest.strip()), bullet, bulletText=f"{num}."))
            i += 1
        elif stripped.startswith("```"):
            # code block — gather until closing fence
            flush()
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            code_style = ParagraphStyle("code", parent=body, fontName="Courier",
                                        fontSize=8.5, leading=10.5, leftIndent=12,
                                        backColor=colors.whitesmoke, spaceAfter=4)
            flow.append(Paragraph("<br/>".join(l.replace(" ", "&nbsp;").replace("&", "&amp;") for l in code_lines), code_style))
        elif stripped == "---":
            flush()
            flow.append(Spacer(1, 4))
            i += 1
        elif stripped == "":
            flush()
            i += 1
        else:
            buf.append(stripped)
            i += 1
    flush()

    doc.build(flow)
    print("Wrote", PDF)


if __name__ == "__main__":
    render()
