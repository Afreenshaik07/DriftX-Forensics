from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)


def generate_incident_pdf(incident: dict[str, Any]) -> bytes:
    """Generate a downloadable PDF report from an incident record."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="DriftX-Forensics Incident Report",
        author="DriftX-Forensics",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        textColor=colors.HexColor("#526783"),
        fontSize=9,
        leading=13,
    ))

    diagnosis = incident.get("diagnosis") or {}
    remediation = incident.get("remediation") or {}
    gate = incident.get("safety_gate") or {}

    story = [
        Paragraph("DriftX-Forensics", styles["Title"]),
        Paragraph("Cross-Stage Drift Investigation Report", styles["ReportSubtitle"]),
        Spacer(1, 5 * mm),
        Paragraph(
            "Generated UTC: " + datetime.now(timezone.utc).isoformat(),
            styles["Normal"],
        ),
        Spacer(1, 5 * mm),
        Paragraph("1. Incident Summary", styles["Heading2"]),
    ]

    summary = [
        ["Field", "Recorded result"],
        ["Scenario", str(incident.get("scenario", "Unspecified"))],
        ["Predicted origin", str(diagnosis.get("predicted_origin", "unknown"))],
        ["Diagnostic confidence", str(diagnosis.get("confidence", "N/A"))],
        ["Propagation path", " -> ".join(diagnosis.get("propagation_path", [])) or "Not established"],
        ["Remediation action", str(remediation.get("action", "N/A"))],
        ["Safety-gate decision", str(gate.get("decision", "NOT_EVALUATED"))],
        ["Deployment authorized", str(incident.get("deployment_authorized", False))],
    ]

    table = Table(summary, colWidths=[48 * mm, 112 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D3E0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white, colors.HexColor("#F1F5FA")
        ]),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([table, Spacer(1, 5 * mm)])

    story.append(Paragraph("2. Forensic Evidence", styles["Heading2"]))
    ranking = diagnosis.get("ranking") or []
    if ranking:
        rows = [["Stage", "Local evidence", "Temporal", "Forensic score"]]
        for item in ranking:
            rows.append([
                str(item.get("stage", "")),
                str(item.get("local_evidence", "")),
                str(item.get("temporal_precedence", "")),
                str(item.get("forensic_score", "")),
            ])
        evidence_table = Table(
            rows, colWidths=[38 * mm, 38 * mm, 38 * mm, 46 * mm], repeatRows=1
        )
        evidence_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F1")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(evidence_table)
    else:
        story.append(Paragraph("Detailed stage ranking was not supplied.", styles["Normal"]))

    story.extend([
        Spacer(1, 4 * mm),
        Paragraph("3. Remediation Recommendation", styles["Heading2"]),
        Paragraph(str(remediation.get("title", "No recommendation supplied")), styles["Normal"]),
        Spacer(1, 2 * mm),
        Paragraph(str(remediation.get("reason", "")), styles["Normal"]),
    ])

    steps = remediation.get("steps") or []
    if steps:
        story.append(Paragraph("Recommended steps", styles["Heading3"]))
        for step in steps:
            story.append(Paragraph("- " + str(step), styles["Normal"]))

    story.extend([
        Spacer(1, 3 * mm),
        Paragraph("4. Safety-Gate Assessment", styles["Heading2"]),
        Paragraph(str(gate.get("reason", "Safety gate was not evaluated.")), styles["Normal"]),
        Spacer(1, 3 * mm),
        Paragraph(
            "Deployment is not executed by this report generator. A PROMOTE result "
            "is an evaluation outcome, not proof that deployment occurred.",
            styles["ReportSubtitle"],
        ),
    ])

    document.build(story)
    return buffer.getvalue()
