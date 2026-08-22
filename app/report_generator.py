from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image
)

import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.colors import darkblue
from reportlab.lib.units import inch
from reportlab.lib import colors

from class_labels import CLASS_LABELS

import os
from datetime import datetime


def generate_pdf_report(
    sample,
    predicted_class,
    confidence,
    probabilities,
    statistics,
    filename
):

    os.makedirs("reports", exist_ok=True)

    pdf_path = os.path.join(
        "reports",
        f"{filename}_inspection_report.pdf"
    )

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    title_style = styles["Heading1"]
    title_style.alignment = TA_CENTER
    title_style.textColor = darkblue

    story = []

    story.append(
        Paragraph(
            "Bridge Structural Health Monitoring",
            title_style
        )
    )

    story.append(
        Paragraph(
            "<b>Inspection Report</b>",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1,0.1*inch))
    story.append(HRFlowable(width="100%", thickness=1, color=darkblue))
    story.append(Spacer(1, 0.2 * inch))
    
    info_table = Table([
        ["Inspection Date", datetime.now().strftime("%d-%m-%Y %H:%M:%S")],
        ["Uploaded File", filename],
        ["Predicted Class", predicted_class],
        ["Confidence", f"{confidence:.2f}%"]
    ], colWidths=[150, 260])

    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.lightgrey),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.black),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))

    story.append(info_table)
    story.append(Spacer(1, 0.25 * inch))


    status_table = Table([
        ["BRIDGE STATUS"],
        [predicted_class]
    ])

    status_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.darkgreen),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("BACKGROUND", (0,1), (-1,1), colors.lightgreen),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,0), 8),
    ]))

    story.append(status_table)

    story.append(Spacer(1, 20))

    story.append(Spacer(1, 0.25 * inch))

    story.append(
        Paragraph(
            "<b>Signal Statistics</b>",
            styles["Heading2"]
        )
    )

    story.append(
        Paragraph(
            f"Minimum : {statistics['Minimum']:.4f}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Maximum : {statistics['Maximum']:.4f}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Mean : {statistics['Mean']:.4f}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"Standard Deviation : {statistics['Standard Deviation']:.4f}",
            styles["BodyText"]
        )
    )

    story.append(
        Paragraph(
            f"RMS : {statistics['RMS']:.4f}",
            styles["BodyText"]
        )
    )

    story.append(Spacer(1, 0.3 * inch))

    # ============================================
# Dynamic Engineering Recommendation
# ============================================

    if predicted_class == "Healthy Bridge (Reference Condition)":
        recommendation = [
            "The bridge is currently classified as Healthy Bridge (Reference Condition).",
            "No significant structural abnormalities were indicated by the vibration-based assessment.",
            "Routine inspection and periodic structural health monitoring are recommended."
        ]

    elif predicted_class == "Baseline After Pier System Installation":
        recommendation = [
            "The bridge is classified as Baseline After Pier System Installation.",
            "This condition represents the post-installation reference state of the pier system.",
            "Continued structural health monitoring is recommended to establish the baseline response."
        ]

    elif predicted_class == "Pier Settlement - 20 mm":
        recommendation = [
            "The assessment indicates pier settlement of approximately 20 mm.",
            "The affected pier and foundation should be inspected for settlement-related effects.",
            "Continued monitoring is recommended to identify any progression."
        ]

    elif predicted_class == "Pier Settlement - 40 mm":
        recommendation = [
            "The assessment indicates pier settlement of approximately 40 mm.",
            "A detailed inspection of the affected pier and its foundation is recommended.",
            "Further monitoring should be performed to determine whether the settlement is progressive."
        ]

    elif predicted_class == "Pier Settlement - 80 mm":
        recommendation = [
            "The assessment indicates significant pier settlement of approximately 80 mm.",
            "A detailed structural and foundation investigation is recommended.",
            "The affected area should be closely monitored and appropriate corrective measures considered."
        ]

    elif predicted_class == "Pier Settlement - 95 mm":
        recommendation = [
            "The assessment indicates severe pier settlement of approximately 95 mm.",
            "Immediate detailed structural and foundation assessment is recommended.",
            "Further engineering evaluation should determine the appropriate corrective action."
        ]

    elif predicted_class == "Foundation Tilt":
        recommendation = [
            "The assessment indicates a foundation tilt condition.",
            "The affected foundation and supporting structural elements should undergo detailed inspection.",
            "Continued monitoring is recommended to determine whether the condition is stable or progressive."
        ]

    elif predicted_class == "Restored Reference Condition":
        recommendation = [
            "The bridge is classified as Restored Reference Condition.",
            "The structural response corresponds to a restored reference state.",
            "Routine inspection and continued structural health monitoring are recommended."
        ]

    elif predicted_class == "Concrete Spalling - 12 m²":
        recommendation = [
            "The assessment indicates concrete spalling affecting approximately 12 m².",
            "The affected concrete should be inspected for deterioration and its underlying causes.",
            "Repair and continued monitoring should be considered based on inspection findings."
        ]

    elif predicted_class == "Concrete Spalling - 24 m²":
        recommendation = [
            "The assessment indicates significant concrete spalling affecting approximately 24 m².",
            "A detailed inspection of the affected structural components is recommended.",
            "Repair measures and continued structural monitoring should be considered."
        ]

    elif predicted_class == "Landslide at Abutment":
        recommendation = [
            "The assessment indicates a landslide condition at the abutment.",
            "The abutment, surrounding ground, drainage conditions, and slope stability should be inspected in detail.",
            "Immediate engineering evaluation is recommended."
        ]

    elif predicted_class == "Concrete Hinge Failure":
        recommendation = [
            "The assessment indicates a concrete hinge failure condition.",
            "A detailed structural inspection of the affected connection and surrounding members is recommended.",
            "Further engineering assessment should determine the required corrective measures."
        ]

    elif predicted_class == "Failure of 2 Anchor Heads":
        recommendation = [
            "The assessment indicates failure of 2 anchor heads.",
            "The affected anchorage system should undergo detailed inspection.",
            "The condition should be evaluated for its effect on structural capacity and appropriate corrective action."
        ]

    elif predicted_class == "Failure of 4 Anchor Heads":
        recommendation = [
            "The assessment indicates failure of 4 anchor heads.",
            "A detailed inspection of the anchorage system is strongly recommended.",
            "Further structural evaluation should determine the impact on load-carrying capacity."
        ]

    elif predicted_class == "Failure of 2 Prestressing Tendons":
        recommendation = [
            "The assessment indicates failure of 2 prestressing tendons.",
            "The affected prestressing system should be inspected in detail.",
            "Further structural assessment is recommended to evaluate the effect on structural performance."
        ]

    elif predicted_class == "Failure of 4 Prestressing Tendons":
        recommendation = [
            "The assessment indicates failure of 4 prestressing tendons.",
            "A detailed structural assessment of the affected prestressed member is recommended.",
            "The remaining prestressing system should also be evaluated."
        ]

    elif predicted_class == "Failure of 6 Prestressing Tendons":
        recommendation = [
            "The assessment indicates failure of 6 prestressing tendons.",
            "This condition requires detailed structural investigation and engineering evaluation.",
            "The effect on load-carrying capacity and structural safety should be assessed."
        ]

    else:
        recommendation = [
            f"The bridge has been classified as: {predicted_class}.",
            "Further engineering investigation is recommended before making maintenance decisions.",
            "Additional structural health monitoring and inspection should be scheduled."
        ]

    for line in recommendation:
        story.append(
            Paragraph(
                line,
                styles["Normal"]
            )
        )

    from reportlab.platypus import PageBreak

    # ============================================
    # Prediction Probability Chart
    # ============================================

    top_indices = np.argsort(probabilities)[::-1][:5]

    top_probs = probabilities[top_indices] * 100

    top_labels = [CLASS_LABELS[i] for i in top_indices]

    plt.figure(figsize=(7, 3.5))
    plt.barh(top_labels[::-1], top_probs[::-1])

    plt.xlabel("Probability (%)")
    plt.title("Top 5 Prediction Probabilities")

    plt.xlim(0, 100)

    chart_path = os.path.join("reports", "probability_chart.png")

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()

    story.append(Spacer(1, 0.25 * inch))

    story.append(PageBreak())
    
    story.append(
        Paragraph(
            "<b>Prediction Probability Distribution</b>",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 0.15 * inch))

    chart = Image(chart_path)
    chart.drawWidth = 7.0 * inch
    chart.drawHeight = 4.5 * inch

    story.append(chart)

    story.append(Spacer(1, 0.25 * inch))

    story.append(
        Paragraph(
            "<b>Top 5 Prediction Probabilities</b>",
            styles["Heading2"]
        )
    )

    story.append(Spacer(1, 0.1 * inch))

    probability_table_data = [
        ["Rank", "Damage Class", "Probability"]
    ]

    for rank, (label, probability) in enumerate(
        zip(top_labels, top_probs),
        start=1
    ):
        probability_table_data.append([
            str(rank),
            label,
            f"{probability:.2f}%"
        ])

    probability_table = Table(
        probability_table_data,
        colWidths=[0.6 * inch, 4.8 * inch, 1.1 * inch]
    )

    probability_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (2, 1), (2, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
        ])
    )

    story.append(probability_table)

    story.append(Spacer(1, 0.25 * inch))

    doc.build(story)

    return pdf_path