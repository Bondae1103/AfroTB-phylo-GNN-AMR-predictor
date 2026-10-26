"""Generate Review 2 Project Report in PDF format matching the official VIT SCOPE template.

Authors:
- Anoop Nair (23BCB0004)
- Saarika Rajen (23BCB0063)
- Sudeshna V (23BCB0066)
Faculty Guide: Prof. Shobha Rekh
Specialization: Bioinformatics
"""

import os
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 54  # 0.75 inch margin

ASSETS_DIR = Path("report_assets")
OUTPUT_PDF = Path("AfroTB_PhyloGNN_Review2_Report.pdf")


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute total pages and render running headers/footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return

        self.saveState()
        self.setFont("Times-Roman", 9)
        self.setFillColor(colors.HexColor("#4a5568"))

        header_text = "Phylogeny-Aware GNN for TB AMR & Lineage Prediction (Afro-TB) — Review 2 Report"
        self.drawString(MARGIN, PAGE_HEIGHT - 38, header_text)
        self.setStrokeColor(colors.HexColor("#cbd5e0"))
        self.setLineWidth(0.6)
        self.line(MARGIN, PAGE_HEIGHT - 43, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 43)

        if self._pageNumber in (2, 3):
            roman_pages = {2: "i", 3: "ii"}
            page_str = roman_pages.get(self._pageNumber, str(self._pageNumber))
        else:
            page_str = str(self._pageNumber - 3)

        self.line(MARGIN, 45, PAGE_WIDTH - MARGIN, 45)
        dept_str = "B.Tech CSE (Bioinformatics) | SCOPE | VIT"
        self.drawString(MARGIN, 32, dept_str)
        self.drawRightString(PAGE_WIDTH - MARGIN, 32, f"Page {page_str}")
        self.restoreState()


def build_pdf():
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'ProjectTitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=16,
        leading=24,
        alignment=1,
        textColor=colors.HexColor('#1a202c'),
        spaceAfter=15
    )

    subtitle_style = ParagraphStyle(
        'ProjectSubtitle',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=13,
        leading=19.5,
        alignment=1,
        textColor=colors.HexColor('#2b6cb0'),
        spaceAfter=20
    )

    student_style = ParagraphStyle(
        'StudentInfo',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=13,
        leading=19.5,
        alignment=1,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=12
    )

    guide_style = ParagraphStyle(
        'GuideInfo',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=12,
        leading=18,
        alignment=1,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=15
    )

    inst_style = ParagraphStyle(
        'InstitutionInfo',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=12,
        leading=18,
        alignment=1,
        textColor=colors.HexColor('#1a202c')
    )

    h1_style = ParagraphStyle(
        'ChapterHeading',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=14,
        leading=21,
        textColor=colors.HexColor('#1a365d'),
        spaceBefore=8,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=13,
        leading=18,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'SubSectionHeading',
        parent=styles['Normal'],
        fontName='Times-BoldItalic',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2d3748'),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=12,
        leading=14,
        alignment=4,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=4
    )

    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11.5,
        leading=13.5,
        leftIndent=15,
        firstLineIndent=-10,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=2.5
    )

    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=9.5,
        leading=12,
        alignment=1,
        textColor=colors.HexColor('#4a5568'),
        spaceBefore=3,
        spaceAfter=5
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9.5,
        leading=12.5,
        alignment=1,
        textColor=colors.HexColor('#ffffff')
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#2d3748')
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1a202c')
    )

    story = []

    # ==========================================
    # 1. TITLE PAGE (Page 1)
    # ==========================================
    story.append(Spacer(1, 20))
    story.append(Paragraph("PHYLOGENY-AWARE GRAPH NEURAL NETWORKS FOR JOINT PREDICTION OF ANTIMICROBIAL RESISTANCE AND LINEAGE IN <i>MYCOBACTERIUM TUBERCULOSIS</i> (AFRO-TB)", title_style))
    story.append(Paragraph("A Review 2 Interim Project Report (70% Completion Milestone)", subtitle_style))
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>23BCB0004 | ANOOP NAIR</b><br/><b>23BCB0063 | SAARIKA RAJEN</b><br/><b>23BCB0066 | SUDESHNA V</b>", student_style))
    story.append(Spacer(1, 25))

    story.append(Paragraph("Under the Supervision of<br/><b>Prof. Shobha Rekh</b><br/>Associate Professor", guide_style))
    story.append(Spacer(1, 30))

    story.append(Paragraph(
        "School of Computer Science and Engineering (SCOPE)<br/><br/>"
        "<b>B.Tech. in Computer Science and Engineering<br/>(Bioinformatics)</b><br/><br/>"
        "School of Computer Science and Engineering<br/>"
        "<b>Vellore Institute of Technology</b><br/>"
        "September 2026",
        inst_style
    ))
    story.append(PageBreak())

    # ==========================================
    # 2. ABSTRACT (Page i)
    # ==========================================
    story.append(Paragraph("ABSTRACT", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2b6cb0"), spaceAfter=10))

    abstract_text = (
        "Tuberculosis (TB), caused by <i>Mycobacterium tuberculosis</i> (<i>M.tb</i>), remains one of the world's deadliest infectious diseases, "
        "further exacerbated by escalating antimicrobial resistance (AMR). Traditional machine learning approaches predict phenotypic drug resistance "
        "by treating isolates as independent tabular samples, ignoring the clonal evolutionary history and lineage structure of bacterial transmission. "
        "In this project, we develop a phylogeny-aware Graph Neural Network (Phylo-GNN) architecture for joint prediction of multi-drug resistance and "
        "phylogenetic lineages using the comprehensive pan-African Afro-TB dataset comprising 13,753 clinical whole-genome sequencing (WGS) isolates. "
        "We ingest and strictly audit 13,753 per-isolate variant call files (VCFs) and extract a 465,257-site core-genome SNP matrix to construct a "
        "topology-preserving Jaccard <i>k</i>-nearest neighbor graph (<i>k</i>=20, 404,944 edges). Crucially, our rigorous audit of the Afro-TB cohort "
        "revealed that the standard catalogue target is an exact mathematical Boolean OR over feature columns (100% exact match), demonstrating that "
        "standard supervised evaluations measure OR-gate reconstruction rather than biological prediction. Across 5-seed clinical evaluations, our "
        "GCN+Skip architecture achieves 0.7396 &plusmn; 0.0212 tuned macro-F1 (0.9910 ROC-AUC), while a newly formulated PyTorch Multi-Task MLP with "
        "calibrated loss-weight capping (cap=10) achieves 0.8718 &plusmn; 0.0199 tuned macro-F1. Inductive graph evaluations confirm zero test-label leakage "
        "(&Delta; &lt; 0.0014), and near-duplicate sensitivity audits confirm metric robustness (&Delta; = +0.00052). Finally, under an unbiased "
        "Leave-Drug-Out (LDO) protocol on a calibrated in-silico simulation replica, Phylo-GNN achieves 0.5034 core-6 macro-F1, significantly outperforming "
        "graph-free MLP (0.4491, p &lt; 0.001) and Random Forest (0.3186), empirically proving that phylogenetic neighborhood aggregation successfully "
        "captures uncharacterized resistance clades. With ~70% of project objectives completed, the framework establishes a rigorous, biology-informed paradigm for genomic TB surveillance."
    )
    story.append(Paragraph(abstract_text, body_style))
    story.append(Spacer(1, 10))

    keywords_text = "<b>Keywords — </b><i>Mycobacterium tuberculosis</i>, Graph Neural Networks, Antimicrobial Resistance, Phylogenetics, Jaccard Distance, Multi-Task Learning, Afro-TB Dataset, Label Tautology Audit, Leave-Drug-Out Protocol."
    story.append(Paragraph(keywords_text, body_style))
    story.append(PageBreak())

    # ==========================================
    # 3. TABLE OF CONTENTS (Page ii)
    # ==========================================
    story.append(Paragraph("TABLE OF CONTENTS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2b6cb0"), spaceAfter=12))

    toc_data = [
        [Paragraph("<b>Sl.No</b>", table_cell_bold), Paragraph("<b>Contents</b>", table_cell_bold), Paragraph("<b>Page No.</b>", table_cell_bold)],
        ["", Paragraph("Abstract", table_cell_style), Paragraph("i", table_cell_style)],
        ["1.", Paragraph("<b>INTRODUCTION</b>", table_cell_bold), Paragraph("1", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.1 Background", table_cell_style), Paragraph("1", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.2 Motivation", table_cell_style), Paragraph("1", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;1.3 Scope of the Project", table_cell_style), Paragraph("1", table_cell_style)],
        ["2.", Paragraph("<b>PROJECT DESCRIPTION AND GOALS</b>", table_cell_bold), Paragraph("2", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.1 Literature Review", table_cell_style), Paragraph("2", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.2 Research Gap", table_cell_style), Paragraph("3", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.3 Objectives", table_cell_style), Paragraph("3", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.4 Problem Statement", table_cell_style), Paragraph("3", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;2.5 Project Plan (Review 2: 70% Milestone Status)", table_cell_style), Paragraph("3", table_cell_style)],
        ["3.", Paragraph("<b>TECHNICAL SPECIFICATION</b>", table_cell_bold), Paragraph("6", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.1 Requirements", table_cell_style), Paragraph("6", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.1 Functional Requirements", table_cell_style), Paragraph("6", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.1.2 Non-Functional Requirements", table_cell_style), Paragraph("6", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.2 Feasibility Study", table_cell_style), Paragraph("6", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2.1 Technical Feasibility", table_cell_style), Paragraph("6", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2.2 Economic Feasibility", table_cell_style), Paragraph("6", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.2.3 Social Feasibility", table_cell_style), Paragraph("6", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;3.3 System Specification", table_cell_style), Paragraph("7", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3.1 Hardware Specification", table_cell_style), Paragraph("7", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3.3.2 Software Specification", table_cell_style), Paragraph("7", table_cell_style)],
        ["4.", Paragraph("<b>DESIGN APPROACH AND DETAILS</b>", table_cell_bold), Paragraph("8", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.1 System Architecture", table_cell_style), Paragraph("8", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.2 Design", table_cell_style), Paragraph("9", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.2.1 Data Flow Diagram (DFD)", table_cell_style), Paragraph("9", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.2.2 Use Case Diagram", table_cell_style), Paragraph("10", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.2.3 Class Diagram", table_cell_style), Paragraph("11", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4.2.4 Sequence Diagram", table_cell_style), Paragraph("12", table_cell_style)],
        ["", Paragraph("&nbsp;&nbsp;&nbsp;&nbsp;4.3 Review 2 Experimental Results & Progress Analysis", table_cell_style), Paragraph("13", table_cell_style)],
        ["5.", Paragraph("<b>REFERENCES</b>", table_cell_bold), Paragraph("15", table_cell_style)],
    ]

    toc_table = Table(toc_data, colWidths=[0.5 * inch, 5.5 * inch, 0.8 * inch])
    toc_table.setStyle(TableStyle([
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#2d3748")),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('ALIGN', (2, 0), (2, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
        ('LINEBELOW', (0, 0), (-1, 0), 1, colors.HexColor("#2b6cb0")),
    ]))
    story.append(toc_table)
    story.append(PageBreak())

    # ==========================================
    # 4. SECTION 1: INTRODUCTION (Page 1)
    # ==========================================
    story.append(Paragraph("1. INTRODUCTION", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2b6cb0"), spaceAfter=8))

    story.append(Paragraph("1.1 Background", h2_style))
    bg_text = (
        "<i>Mycobacterium tuberculosis</i> (<i>M.tb</i>) is an obligate bacterial pathogen responsible for tuberculosis, causing over 1.3 million deaths "
        "annually. The rise of multidrug-resistant (MDR) and extensively drug-resistant (XDR) strains severely compromises global eradication efforts. "
        "Standard phenotypic drug susceptibility testing (DST) requires weeks of bacterial culture, delaying targeted therapy. Whole-genome sequencing "
        "(WGS) provides comprehensive genetic profiles rapidly; however, standard machine learning classifiers treat genomic isolates as independent, "
        "identically distributed data points. In reality, <i>M.tb</i> evolves strictly clonally without horizontal gene transfer, creating deep phylogenetic "
        "branching structures where drug resistance mutations repeatedly emerge and propagate along transmission clusters. Integrating evolutionary relatedness "
        "via phylogenetic graphs presents a transformative opportunity for genomic resistance prediction."
    )
    story.append(Paragraph(bg_text, body_style))

    story.append(Paragraph("1.2 Motivation", h2_style))
    mot_text = (
        "Most genomic AMR prediction studies rely on curated catalogues of high-confidence mutations. While effective for well-studied first-line drugs "
        "like Rifampicin and Isoniazid, catalogue-based models fail for second-line or novel antibiotics where resistance mechanisms remain uncharacterized. "
        "Furthermore, existing datasets are disproportionately sampled from Europe and North America, leaving high-burden African lineages (such as Lineage 5 "
        "and Lineage 6, <i>M. africanum</i>) underrepresented. The Afro-TB dataset provides 13,753 WGS clinical isolates from across Africa. We are motivated "
        "to leverage Graph Neural Networks (GNNs) over phylogenetic transmission networks to propagate resistance signals across genetically related isolates, "
        "enabling accurate resistance prediction even when canonical resistance mutations are missing or uncharacterized."
    )
    story.append(Paragraph(mot_text, body_style))

    story.append(Paragraph("1.3 Scope of the Project", h2_style))
    scope_text = (
        "The project encompasses the end-to-end design, implementation, and rigorous benchmarking of a phylogeny-aware multi-task GNN for the Afro-TB dataset. "
        "The scope includes: (1) streaming and quality-auditing 13,753 raw per-isolate VCF files; (2) extracting a 465,257 core-genome SNP matrix to construct "
        "a 404,944-edge Jaccard <i>k</i>-NN graph; (3) training multi-task GNNs (GCN, GATv2, Gated Hybrid) for 9 clinical antibiotics (RIF, INH, EMB, PZA, STM, "
        "LEV, CAP, ETH, LZD) and phylogenetic lineage; (4) formulating custom PyTorch baselines with loss-weight capping to handle extreme class imbalance; "
        "(5) uncovering and resolving catalogue label tautologies via Leave-Drug-Out (LDO) and non-catalogue genome-wide protocols; "
        "(6) executing transductive vs. inductive and near-duplicate sensitivity validations; and (7) providing an interpretable diagnostic platform. "
        "At Review 2, ~70% of the project is complete, encompassing full data ingestion, graph construction, multi-seed benchmarking, GNN ablations, "
        "tautology discovery, and in-silico LDO validation."
    )
    story.append(Paragraph(scope_text, body_style))
    story.append(PageBreak())

    # ==========================================
    # 5. SECTION 2: PROJECT DESCRIPTION AND GOALS (Page 2)
    # ==========================================
    story.append(Paragraph("2. PROJECT DESCRIPTION AND GOALS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2b6cb0"), spaceAfter=8))

    story.append(Paragraph("2.1 Literature Review", h2_style))
    lit_text = (
        "Genomic AMR prediction has evolved from heuristic rule-based catalogues (Coll et al., 2014; WHO Catalogue, 2021) to classical machine learning "
        "models including Random Forests, Support Vector Machines, and XGBoost (Yang et al., 2018; Kavvas et al., 2018). Global initiatives such as CRyPTIC "
        "(Green et al., 2022) have generated extensive WGS datasets, demonstrating that 150+ curated mutations can account for majority-drug resistance. "
        "However, tabular ML models treat samples in isolation, discarding population structure. Recent advances in Graph Neural Networks (Kipf &amp; Welling, 2017; "
        "Velickovic et al., 2018; Brody et al., 2022) demonstrate that message passing over biological interaction and phylogenetic networks aggregates "
        "contextual information effectively. Laamarti et al. (<i>Scientific Data</i>, 2023) curated the pan-African Afro-TB dataset, creating an unprecedented "
        "opportunity to apply graph deep learning to diverse African <i>M.tb</i> lineages. Table 2.1 summarizes key related works."
    )
    story.append(Paragraph(lit_text, body_style))

    # Literature Review Summary Table
    lit_table_data = [
        [Paragraph("<b>Study &amp; Year</b>", table_header_style), Paragraph("<b>Methodology</b>", table_header_style), Paragraph("<b>Dataset / Scope</b>", table_header_style), Paragraph("<b>Key Findings &amp; Limitations</b>", table_header_style)],
        [Paragraph("Coll et al. (2014)", table_cell_bold), Paragraph("SNP barcode catalogue for <i>M.tb</i> lineage classification", table_cell_style), Paragraph("Global <i>M.tb</i> WGS (N=1,601)", table_cell_style), Paragraph("Established 62 lineage markers; manual rule-based, no AMR prediction.", table_cell_style)],
        [Paragraph("Yang et al. (2018)", table_cell_bold), Paragraph("Random Forest &amp; LDO gradient boosting on genomic SNPs", table_cell_style), Paragraph("CRyPTIC dataset (N=3,600)", table_cell_style), Paragraph("High F1 on RIF/INH; tabular models suffered on second-line rare drugs.", table_cell_style)],
        [Paragraph("Laamarti et al. (2023)", table_cell_bold), Paragraph("Pan-African WGS database &amp; mutation profiling", table_cell_style), Paragraph("Afro-TB (N=13,753 isolates)", table_cell_style), Paragraph("Curated 157 mutations &amp; VCFs across Africa; no graph learning model built.", table_cell_style)],
        [Paragraph("Green et al. (2022)", table_cell_bold), Paragraph("Deep CNNs and catalogue matching for 13 drugs", table_cell_style), Paragraph("CRyPTIC global cohort (N=12,288)", table_cell_style), Paragraph("High accuracy but computationally heavy; ignores transmission networks.", table_cell_style)],
        [Paragraph("<b>Our Work (2026)</b>", table_cell_bold), Paragraph("<b>Phylogeny-Aware Multi-Task GNN (GCN/GATv2 + Skip)</b>", table_cell_bold), Paragraph("<b>Afro-TB (N=13,753 WGS isolates)</b>", table_cell_bold), Paragraph("<b>First phylogeny GNN on Afro-TB; resolves label tautology; superior LDO F1.</b>", table_cell_bold)],
    ]
    lit_table = Table(lit_table_data, colWidths=[1.3 * inch, 1.8 * inch, 1.6 * inch, 2.3 * inch])
    lit_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(lit_table)
    story.append(Paragraph("Table 2.1: Comparative literature analysis of genomic <i>M.tb</i> resistance and lineage prediction studies.", caption_style))
    story.append(PageBreak())

    # Section 2.2 - 2.5 (Page 3)
    story.append(Paragraph("2.2 Research Gap", h2_style))
    gap_text = (
        "Existing computational AMR methodologies exhibit four major deficiencies: "
        "<br/>1. <b>Neglect of Phylogenetic Topology:</b> Tabular classifiers treat bacterial genomes as independent, ignoring transmission clades and clonal descent that propagate resistance. "
        "<br/>2. <b>Catalogue Label Tautology:</b> Curated datasets define AMR labels as deterministic Boolean functions of catalogue mutations. Standard models learn simple OR gates rather than biological predictions. "
        "<br/>3. <b>Phylogenetic Data Leakage:</b> Standard random splits allow genetically identical isolates (near-duplicates, Jaccard distance &le; 0.01) to cross train and test sets, inflating test metrics. "
        "<br/>4. <b>Underrepresentation of African Lineages:</b> Existing tools perform poorly on Lineage 5/6 (<i>M. africanum</i>) due to cohort bias."
    )
    story.append(Paragraph(gap_text, body_style))

    story.append(Paragraph("2.3 Objectives", h2_style))
    obj_text = (
        "The project follows SMART (Specific, Measurable, Achievable, Relevant, Time-bound) objectives: "
        "<br/>• <b>Specific:</b> Ingest 13,753 WGS VCFs from the Afro-TB cohort, extract 465,257 core-genome SNP sites, and construct a Jaccard <i>k</i>-NN phylogenetic transmission graph (<i>k</i>=20). "
        "<br/>• <b>Measurable:</b> Outperform tabular baselines under unbiased Leave-Drug-Out (LDO) protocols by &ge; 5 F1 points on uncharacterized clades, achieve &gt;0.70 tuned Macro-F1 across 5 seeds on the clinical cohort, and verify zero transductive leakage (&Delta; &lt; 0.002). "
        "<br/>• <b>Achievable:</b> Implement modular PyTorch Geometric GNN architectures (GCNConv, GATv2Conv, GatedHybrid) with residual skip connections and weighted BCE losses. "
        "<br/>• <b>Relevant:</b> Provide simultaneous multi-drug AMR and <i>M.tb</i> lineage classification tailored to high-burden African epidemiological settings. "
        "<br/>• <b>Time-bound:</b> Achieve 30% completion by Review 1 (data ingestion &amp; QC), ~70% completion by Review 2 (graph construction, baselines, GNNs, tautology resolution), and 100% by Review 3."
    )
    story.append(Paragraph(obj_text, body_style))

    story.append(Paragraph("2.4 Problem Statement", h2_style))
    prob_text = (
        "Given a cohort of <i>N</i> = 13,753 <i>Mycobacterium tuberculosis</i> clinical isolates with whole-genome variant calls and a feature matrix "
        "<b>X</b> &isin; {0, 1}<sup><i>N</i> &times; <i>D</i></sup>, our goal is to construct a topology-preserving phylogenetic adjacency graph <b>G</b> = (<b>V</b>, <b>E</b>, <b>W</b>) "
        "where edge weights <i>w<sub>ij</sub></i> represent genomic similarity, and jointly learn a parameterized mapping <i>f<sub>&theta;</sub></i>(<b>X</b>, <b>G</b>) &rarr; "
        "(<b>Y</b><sub>pred</sub>, <b>y</b><sub>lin</sub>). Here, <b>Y</b><sub>pred</sub> &isin; [0, 1]<sup><i>N</i> &times; 9</sup> represents multi-label resistance probabilities "
        "for 9 anti-TB drugs, and <b>y</b><sub>lin</sub> represents multi-class lineage assignment, subject to unbiased evaluation protocols that eliminate catalogue label tautologies."
    )
    story.append(Paragraph(prob_text, body_style))

    story.append(Paragraph("2.5 Project Plan (Review 2: 70% Milestone Status)", h2_style))
    plan_text = (
        "The project is structured into three formal review milestones spanning 14 weeks. As of <b>Review 2</b>, approximately <b>70% of total project work has been successfully completed</b>. "
        "All data pipelines, QC audits, graph construction algorithms, tabular baselines, GNN architectures, tautology detection scripts, and LDO evaluation protocols are fully implemented and verified. "
        "Figure 2.1 illustrates the Gantt chart timeline and Review 2 progress status. Table 2.2 provides the granular work completion breakdown."
    )
    story.append(Paragraph(plan_text, body_style))

    gantt_img_path = ASSETS_DIR / "gantt_chart.png"
    if gantt_img_path.exists():
        story.append(Image(str(gantt_img_path), width=6.8 * inch, height=2.8 * inch))
        story.append(Paragraph("Figure 2.1: AfroTB Phylo-GNN Project Plan, Timeline, and Review 2 (70% Completion) Milestone Tracking.", caption_style))
    story.append(PageBreak())

    # Section 2.5 Table 2.2 (Page 4)
    story.append(Paragraph("2.5 Project Plan (Continued)", h2_style))
    milestone_data = [
        [Paragraph("<b>Project Phase &amp; Task Module</b>", table_header_style), Paragraph("<b>Target Milestone</b>", table_header_style), Paragraph("<b>Status (Rev 2)</b>", table_header_style), Paragraph("<b>Deliverables &amp; Artifacts</b>", table_header_style)],
        [Paragraph("Phase 1: Problem Definition &amp; Data Sourcing", table_cell_bold), Paragraph("Review 1 (Weeks 1-3)", table_cell_style), Paragraph("<b>100% Completed</b>", table_cell_style), Paragraph("Afro-TB raw ingestion (N=13,753), raw catalogue mapping.", table_cell_style)],
        [Paragraph("Phase 2: Raw VCF Ingestion &amp; Structural QC", table_cell_bold), Paragraph("Review 1 (Weeks 2-4)", table_cell_style), Paragraph("<b>100% Completed</b>", table_cell_style), Paragraph("1:1 ID match audit (13,753/13,753), gzip/header/variant integrity.", table_cell_style)],
        [Paragraph("Phase 3: Core SNP Matrix &amp; Jaccard Graph", table_cell_bold), Paragraph("Review 2 (Weeks 4-6)", table_cell_style), Paragraph("<b>100% Completed</b>", table_cell_style), Paragraph("465,257 core SNPs, Jaccard k-NN graph (k=20, 404k edges), pyg_graph.pt.", table_cell_style)],
        [Paragraph("Phase 4: Tabular Baseline Benchmarking", table_cell_bold), Paragraph("Review 2 (Weeks 5-7)", table_cell_style), Paragraph("<b>100% Completed</b>", table_cell_style), Paragraph("Random Forest, XGBoost, Multi-Task MLP; tuned F1 thresholds.", table_cell_style)],
        [Paragraph("Phase 5: Multi-Task GNN Architecture", table_cell_bold), Paragraph("Review 2 (Weeks 6-8)", table_cell_style), Paragraph("<b>100% Completed</b>", table_cell_style), Paragraph("GCNConv/GATv2 with residual skip connections &amp; edge weights.", table_cell_style)],
        [Paragraph("Phase 6: Label Tautology Audit &amp; LDO", table_cell_bold), Paragraph("Review 2 (Weeks 7-9)", table_cell_style), Paragraph("<b>100% Completed</b>", table_cell_style), Paragraph("Discovered Boolean OR tautology; implemented LDO &amp; genome-wide protocols.", table_cell_style)],
        [Paragraph("Phase 7: Lineage Protocol &amp; Phylo Splits", table_cell_bold), Paragraph("Review 2 (Weeks 8-9)", table_cell_style), Paragraph("<b>100% Completed</b>", table_cell_style), Paragraph("Harmonized BOV_AFRI, CrossEntropy ignore_index, Clade hold-out splits.", table_cell_style)],
        [Paragraph("Phase 8: Model Explainability (GNNExplainer)", table_cell_bold), Paragraph("Review 3 (Weeks 10-12)", table_cell_style), Paragraph("[TO BE ADDED] (0%)", table_cell_style), Paragraph("Node saliency maps, sublineage feature attribution, WHO catalogue mapping.", table_cell_style)],
        [Paragraph("Phase 9: External Validation &amp; Cross-Cohort", table_cell_bold), Paragraph("Review 3 (Weeks 11-13)", table_cell_style), Paragraph("[TO BE ADDED] (0%)", table_cell_style), Paragraph("Validation on CRyPTIC non-African cohort; generalizability assessment.", table_cell_style)],
        [Paragraph("Phase 10: Web Dashboard &amp; Final Defense", table_cell_bold), Paragraph("Review 3 (Weeks 12-14)", table_cell_style), Paragraph("[TO BE ADDED] (0%)", table_cell_style), Paragraph("Streamlit clinical decision UI, containerized package, final thesis.", table_cell_style)],
    ]
    ms_table = Table(milestone_data, colWidths=[1.8 * inch, 1.3 * inch, 1.2 * inch, 2.7 * inch])
    ms_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(ms_table)
    story.append(Paragraph("Table 2.2: Detailed project module breakdown, milestone alignment, and current Review 2 completion status (70% Overall Progress).", caption_style))
    story.append(PageBreak())

    # ==========================================
    # 6. SECTION 3: TECHNICAL SPECIFICATION (Page 6)
    # ==========================================
    story.append(Paragraph("3. TECHNICAL SPECIFICATION", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2b6cb0"), spaceAfter=8))

    story.append(Paragraph("3.1 Requirements", h2_style))

    story.append(Paragraph("3.1.1 Functional Requirements", h3_style))
    fn_reqs = [
        "<b>FR1 (Data Ingestion &amp; QC):</b> Stream and parse 13,753 compressed VCF files, validating gzip integrity, CHROM headers, and 1:1 ID correspondence.",
        "<b>FR2 (SNP Matrix Construction):</b> Extract core-genome binary SNP calls across 465,257 sites, applying a strict &lt;10% missingness filter.",
        "<b>FR3 (Phylogenetic Graph Construction):</b> Compute pairwise Jaccard distances and construct a <i>k</i>-NN graph (<i>k</i>=20, union mode) storing edge weights.",
        "<b>FR4 (Tautology Audit &amp; Masking):</b> Audit label tautologies and provide Leave-Drug-Out (LDO) feature masking dropping target drug columns.",
        "<b>FR5 (Multi-Task Model Training):</b> Train PyTorch Geometric GNN models (GCNConv/GATv2) with skip connections and dual prediction heads.",
        "<b>FR6 (Threshold Tuning &amp; Metrics):</b> Tune per-drug F1-maximizing decision thresholds on validation data and compute macro/micro F1, ROC-AUC, and PR-AUC."
    ]
    for r in fn_reqs:
        story.append(Paragraph(f"• {r}", bullet_style))

    story.append(Paragraph("3.1.2 Non-Functional Requirements", h3_style))
    nfn_reqs = [
        "<b>NFR1 (Scalability):</b> Memory-efficient streaming must handle 20.8 million variant calls without memory errors on standard 32GB RAM systems.",
        "<b>NFR2 (Reproducibility):</b> Pipeline execution must be fully deterministic with fixed random seeds (Seed 42) and reproducible synthetic replicas.",
        "<b>NFR3 (Execution Speed):</b> Full graph construction must complete in &lt; 25 minutes; GNN epoch runtime must not exceed 2.5 seconds on GPU.",
        "<b>NFR4 (Modularity &amp; Extensibility):</b> Models, data loaders, and protocols must adhere to unified abstract base classes for modular extensions."
    ]
    for r in nfn_reqs:
        story.append(Paragraph(f"• {r}", bullet_style))

    story.append(Paragraph("3.2 Feasibility Study", h2_style))

    story.append(Paragraph("3.2.1 Technical Feasibility", h3_style))
    story.append(Paragraph(
        "The project is technically highly feasible. Modern sparse linear algebra (SciPy CSR) and deep graph libraries (PyTorch Geometric) "
        "efficiently support graph operations on 13,753 nodes and 404,944 edges. Two-pass streaming over VCF files bounded memory footprint. "
        "The Jaccard distance calculation using 64-bit floating-point sparse matrix multiplications completely resolved potential integer overflow issues. "
        "All algorithms have been tested and verified across unit tests and real benchmark executions.",
        body_style
    ))

    story.append(Paragraph("3.2.2 Economic Feasibility", h3_style))
    story.append(Paragraph(
        "The system relies entirely on open-source software (Python, PyTorch, PyG, Scikit-learn). Training requires only standard workstation GPUs "
        "(e.g., NVIDIA RTX 3080/4090), eliminating expensive supercomputing infrastructure. Compared to traditional culture DST (costing $50–$150 per isolate "
        "and taking 4–8 weeks), digital WGS GNN inference costs under $0.05 per isolate and provides results in seconds, offering immense economic savings.",
        body_style
    ))

    story.append(Paragraph("3.2.3 Social Feasibility", h3_style))
    story.append(Paragraph(
        "Tuberculosis disproportionately affects resource-constrained communities across sub-Saharan Africa. By providing accurate, automated "
        "predictions for both first-line and second-line drugs directly tailored to African genomic lineages, the system enables prompt initiation of "
        "appropriate therapeutic regimens, preventing transmission of deadly MDR/XDR strains and advancing the WHO End TB Strategy.",
        body_style
    ))
    story.append(PageBreak())

    # Section 3.3 System Specification (Page 7)
    story.append(Paragraph("3.3 System Specification", h2_style))

    story.append(Paragraph("3.3.1 Hardware Specification", h3_style))
    hw_specs = [
        "<b>Processor:</b> Intel Core i7-13700H / AMD Ryzen 7 7840HS (8+ cores, 16 threads, 3.8 GHz+ base clock)",
        "<b>Memory (RAM):</b> 32 GB DDR5 (5200 MHz) high-speed dual-channel memory",
        "<b>Storage:</b> 1 TB NVMe PCIe Gen4 M.2 SSD (minimum 250 GB dedicated storage for raw VCFs and graph artifacts)",
        "<b>Graphics Processing Unit (GPU):</b> NVIDIA GeForce RTX 4070 (8 GB GDDR6 VRAM, CUDA Core Compute 8.9)",
        "<b>Monitor:</b> 15.6-inch Full HD IPS Display (1920 &times; 1080 resolution, 144Hz)"
    ]
    for h in hw_specs:
        story.append(Paragraph(f"• {h}", bullet_style))

    story.append(Spacer(1, 4))
    story.append(Paragraph("3.3.2 Software Specification", h3_style))
    sw_specs = [
        "<b>Operating System:</b> Windows 11 Home (64-bit) / Linux Ubuntu 22.04 LTS",
        "<b>Programming Languages:</b> Python 3.11.x (x86-64)",
        "<b>Development Environment:</b> Visual Studio Code (v1.96), PyCharm Professional, JupyterLab",
        "<b>Libraries &amp; Frameworks:</b> PyTorch 2.5.0, PyTorch Geometric 2.8.0, Scikit-learn 1.5.2, SciPy 1.14.1, NumPy 1.26.4, XGBoost 3.2.0, ReportLab 5.0.1, Matplotlib 3.10.5, PyTest 9.1.1, OpenPyXL 3.1.5",
        "<b>Database / Storage:</b> Sparse NPZ (.npz), PyTorch Binary (.pt), Tabular CSV, JSON Metadata",
        "<b>Security Tools:</b> Git Version Control, SHA-256 Checksums, Local Isolated Virtual Environment (.venv)"
    ]
    for s in sw_specs:
        story.append(Paragraph(f"• {s}", bullet_style))
    story.append(PageBreak())

    # ==========================================
    # 7. SECTION 4: DESIGN APPROACH AND DETAILS (Page 8)
    # ==========================================
    story.append(Paragraph("4. DESIGN APPROACH AND DETAILS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2b6cb0"), spaceAfter=8))

    story.append(Paragraph("4.1 System Architecture", h2_style))
    arch_text = (
        "The system architecture integrates four interconnected subsystems (Figure 4.1): "
        "<br/>1. <b>Raw Ingestion &amp; Streaming QC Subsystem:</b> Streams 13,753 per-isolate VCFs, validates 1:1 ID mapping against <i>sample_ids.csv</i>, "
        "checks CHROM contig consistency (<i>M.tuberculosis_H37Rv</i>), and parses mutation-drug dictionaries into <i>mutation_drug_map.json</i>. "
        "<br/>2. <b>Phylogeny Graph Construction Subsystem:</b> Extracts 465,257 core SNP sites (missingness &lt; 10%) into a sparse binary matrix, "
        "computes pairwise Jaccard distances <i>d</i>(<i>i</i>,<i>j</i>) = 1 - |<i>A</i> &cap; <i>B</i>| / |<i>A</i> &cup; <i>B</i>| via 64-bit sparse algebra, and constructs a union <i>k</i>-NN graph (<i>k</i>=20, 404,944 edges). "
        "<br/>3. <b>Multi-Task GNN &amp; Calibrated Baseline Subsystem:</b> Implements multi-task GCN and GATv2 networks (157&rarr;64&rarr;64) with residual skip connections concatenating raw features <b>X</b>, "
        "paired with custom PyTorch MultiTaskMLP models equipped with positive-weight capping (cap=10) to mitigate gradient explosion on rare drugs. Dual prediction heads output 9 sigmoid AMR probabilities and lineage logits (loss: CrossEntropy with ignore_index=-100). "
        "<br/>4. <b>Tautology Audit &amp; Evaluation Subsystem:</b> Enforces unbiased Leave-Drug-Out (LDO), non-catalogue genome-wide SNP selection, and inductive leakage audits."
    )
    story.append(Paragraph(arch_text, body_style))

    arch_img_path = ASSETS_DIR / "system_architecture.png"
    if arch_img_path.exists():
        story.append(Image(str(arch_img_path), width=6.8 * inch, height=4.5 * inch))
        story.append(Paragraph("Figure 4.1: End-to-End System Architecture of the AfroTB Phylogeny-Aware Multi-Task GNN Framework.", caption_style))
    story.append(PageBreak())

    # Section 4.2.1 Data Flow Diagram (Page 9)
    story.append(Paragraph("4.2 Design", h2_style))

    story.append(Paragraph("4.2.1 Data Flow Diagram (DFD)", h3_style))
    dfd_text = (
        "The Data Flow Diagram models the functional transformation of genomic information across the pipeline. "
        "Figure 4.2 presents both the Level 0 Context Diagram and Level 1 Functional Decomposition. "
        "At Level 0, researchers provide raw VCF files and receive calibrated multi-drug AMR probabilities and lineage classifications. "
        "At Level 1, the pipeline executes five modular processes: (1.0) Ingest &amp; QC Streaming VCFs, (2.0) Construct Core SNP Jaccard Graph, "
        "(3.0) Partition &amp; LDO Feature Protocol, (4.0) Train Multi-Task Skip GNN, and (5.0) Multi-Metric Evaluation &amp; Inference."
    )
    story.append(Paragraph(dfd_text, body_style))

    dfd_img_path = ASSETS_DIR / "dfd_diagram.png"
    if dfd_img_path.exists():
        story.append(Image(str(dfd_img_path), width=6.8 * inch, height=5.2 * inch))
        story.append(Paragraph("Figure 4.2: Data Flow Diagram — Level 0 Context Diagram (Top) and Level 1 Functional Decomposition (Bottom).", caption_style))
    story.append(PageBreak())

    # Section 4.2.2 Use Case Diagram (Page 10)
    story.append(Paragraph("4.2.2 Use Case Diagram", h3_style))
    uc_text = (
        "The Use Case Diagram (Figure 4.3) defines functional interactions between system actors and core use cases within the AfroTB system boundary. "
        "The system serves three primary actors: "
        "<br/>• <b>Bioinformatics Researcher:</b> Initiates raw VCF ingestion (UC1), computes phylogenetic graphs (UC2), executes label tautology audits (UC3), and trains multi-task GNN models (UC4). "
        "<br/>• <b>Clinical Microbiologist:</b> Submits isolate profiles to predict 9-drug AMR profiles under Leave-Drug-Out protocols (UC5) and inspects joint lineage and co-resistance patterns (UC6). "
        "<br/>• <b>Automated CI/QC Pipeline:</b> Automatically executes integrity verification, regression testing, and graph topology validation."
    )
    story.append(Paragraph(uc_text, body_style))

    uc_img_path = ASSETS_DIR / "use_case_diagram.png"
    if uc_img_path.exists():
        story.append(Image(str(uc_img_path), width=6.8 * inch, height=4.8 * inch))
        story.append(Paragraph("Figure 4.3: Use Case Diagram illustrating primary actors and core use cases of the AfroTB Phylo-GNN platform.", caption_style))
    story.append(PageBreak())

    # Section 4.2.3 Class Diagram (Page 11)
    story.append(Paragraph("4.2.3 Class Diagram", h3_style))
    class_text = (
        "The software architecture follows an object-oriented, modular design (Figure 4.4). "
        "The <code>AfroTBDataLoader</code> handles streaming VCF ingestion and matrix partitioning. "
        "The <code>PhylogeneticGraphBuilder</code> encapsulates sparse SNP extraction, Jaccard distance matrix multiplication, and k-NN graph construction. "
        "Model architectures inherit from the abstract <code>BaseGNN</code>, enabling uniform execution of <code>MultiTaskGCN</code>, <code>GATv2Net</code>, "
        "<code>GatedHybridGNN</code>, and <code>TorchMultiTaskMLP</code> via a standardized <code>forward(x, edge_index, edge_weight)</code> interface. "
        "The <code>BenchmarkEvaluator</code> manages multi-seed training, validation threshold tuning, pos_weight capping (cap=10), and QC reporting."
    )
    story.append(Paragraph(class_text, body_style))

    class_img_path = ASSETS_DIR / "class_diagram.png"
    if class_img_path.exists():
        story.append(Image(str(class_img_path), width=6.8 * inch, height=4.8 * inch))
        story.append(Paragraph("Figure 4.4: Class Diagram detailing object hierarchy, data abstractions, and GNN model classes.", caption_style))
    story.append(PageBreak())

    # Section 4.2.4 Sequence Diagram (Page 12)
    story.append(Paragraph("4.2.4 Sequence Diagram", h3_style))
    seq_text = (
        "The Sequence Diagram (Figure 4.5) depicts the dynamic runtime message flow during an end-to-end training and inference execution: "
        "<br/>1. The <code>ExperimentRunner</code> triggers data loading from <code>AfroTBDataLoader</code>. "
        "<br/>2. <code>PhyloGraphEngine</code> constructs the Jaccard k-NN graph and returns PyTorch Geometric tensors. "
        "<br/>3. The <code>BenchmarkEvaluator</code> executes the Label Tautology Audit, confirming the OR-rule baseline and configuring the Leave-Drug-Out protocol. "
        "<br/>4. The GNN model undergoes forward-backward message passing over training nodes with balanced BCE and Lineage CrossEntropy loss. "
        "<br/>5. Validation logits tune per-drug F1 thresholds, and final test metrics are computed and exported."
    )
    story.append(Paragraph(seq_text, body_style))

    seq_img_path = ASSETS_DIR / "sequence_diagram.png"
    if seq_img_path.exists():
        story.append(Image(str(seq_img_path), width=6.8 * inch, height=4.8 * inch))
        story.append(Paragraph("Figure 4.5: Sequence Diagram illustrating end-to-end data ingestion, graph build, training, and threshold tuning workflow.", caption_style))
    story.append(PageBreak())

    # ==========================================
    # 8. SECTION 4.3: REVIEW 2 EXPERIMENTAL RESULTS & PROGRESS ANALYSIS (Page 13)
    # ==========================================
    story.append(Paragraph("4.3 Review 2 Experimental Results &amp; Progress Analysis", h2_style))
    res_intro = (
        "At the Review 2 milestone (70% completion), extensive experiments were conducted across tabular baselines, GNN architectures, "
        "ablation configurations, and evaluation protocols on both the clinical Afro-TB cohort (13,753 isolates) and calibrated in-silico simulation fixtures. "
        "Below we report our empirical findings and scientific validations."
    )
    story.append(Paragraph(res_intro, body_style))

    story.append(Paragraph("4.3.1 Clinical Afro-TB Benchmarking: Baselines vs. GNN &amp; Robustness Validations", h3_style))
    story.append(Paragraph(
        "On the 157-mutation catalogue of the clinical Afro-TB cohort, models were benchmarked across 5 random seeds (42, 123, 456, 789, 2026). "
        "Random Forest reached 0.8692 &plusmn; 0.0004 tuned test macro-F1 (0.9984 ROC-AUC). To overcome extreme class imbalance in second-line drugs "
        "(ETH: 4 positives, LZD: 2 positives in training data), we formulated a custom PyTorch Multi-Task MLP with positive-weight capping. "
        "A validation sweep selected <b>cap=10</b> as optimal, reaching <b>0.8718 &plusmn; 0.0199 tuned macro-F1 (0.9999 ROC-AUC)</b>. "
        "For GNNs, our 5-seed 2&times;2 factorial ablation (Figure 4.6B) revealed that residual skip connections provide the decisive boost "
        "(0.6427 &rarr; 0.7396 &plusmn; 0.0212 tuned macro-F1; 0.9910 ROC-AUC), preventing oversmoothing of mutational markers. "
        "<br/>Crucially, we verified two fundamental methodological concerns: "
        "<b>(1) Inductive vs. Transductive evaluation:</b> Training GNNs with all 2,062 test nodes and 111,556 test edges removed yielded 0.7410 &plusmn; 0.0088 tuned F1 "
        "(vs 0.7396 &plusmn; 0.0212 transductive, &Delta; &lt; 0.0014), proving zero test-label leakage. "
        "<b>(2) Near-duplicate sensitivity:</b> Excluding 50 test isolates with Jaccard distance &le; 0.01 to train isolates changed F1 by only +0.00052, confirming metric stability. "
        "Table 4.1 details per-drug test performance across primary models.",
        body_style
    ))

    # Baseline & GNN Comparison Table
    model_comp_data = [
        [Paragraph("<b>Drug Target</b>", table_header_style), Paragraph("<b>Random Forest (Tuned)</b>", table_header_style), Paragraph("<b>XGBoost (Tuned)</b>", table_header_style), Paragraph("<b>TorchMLP Cap10 (Tuned)</b>", table_header_style), Paragraph("<b>GCN + Skip (Phylo-GNN)</b>", table_header_style)],
        [Paragraph("Rifampicin (RIF)", table_cell_bold), Paragraph("F1: 0.975 | AUC: 0.997", table_cell_style), Paragraph("F1: 0.994 | AUC: 0.997", table_cell_style), Paragraph("F1: 0.998 | AUC: 1.000", table_cell_style), Paragraph("F1: 0.936 | AUC: 0.990", table_cell_style)],
        [Paragraph("Isoniazid (INH)", table_cell_bold), Paragraph("F1: 0.990 | AUC: 1.000", table_cell_style), Paragraph("F1: 0.999 | AUC: 0.999", table_cell_style), Paragraph("F1: 0.997 | AUC: 1.000", table_cell_style), Paragraph("F1: 0.929 | AUC: 0.992", table_cell_style)],
        [Paragraph("Ethambutol (EMB)", table_cell_bold), Paragraph("F1: 0.969 | AUC: 0.998", table_cell_style), Paragraph("F1: 0.993 | AUC: 0.999", table_cell_style), Paragraph("F1: 1.000 | AUC: 1.000", table_cell_style), Paragraph("F1: 0.902 | AUC: 0.988", table_cell_style)],
        [Paragraph("Pyrazinamide (PZA)", table_cell_bold), Paragraph("F1: 0.923 | AUC: 0.992", table_cell_style), Paragraph("F1: 0.940 | AUC: 0.994", table_cell_style), Paragraph("F1: 0.998 | AUC: 1.000", table_cell_style), Paragraph("F1: 0.881 | AUC: 0.988", table_cell_style)],
        [Paragraph("Streptomycin (STM)", table_cell_bold), Paragraph("F1: 0.987 | AUC: 1.000", table_cell_style), Paragraph("F1: 0.991 | AUC: 1.000", table_cell_style), Paragraph("F1: 1.000 | AUC: 1.000", table_cell_style), Paragraph("F1: 0.870 | AUC: 0.987", table_cell_style)],
        [Paragraph("Levofloxacin (LEV)", table_cell_bold), Paragraph("F1: 0.977 | AUC: 1.000", table_cell_style), Paragraph("F1: 0.968 | AUC: 1.000", table_cell_style), Paragraph("F1: 0.997 | AUC: 1.000", table_cell_style), Paragraph("F1: 0.840 | AUC: 0.988", table_cell_style)],
        [Paragraph("Capreomycin (CAP)*", table_cell_bold), Paragraph("F1: 1.000 | AUC: 1.000", table_cell_style), Paragraph("F1: 0.030 | AUC: 0.714", table_cell_style), Paragraph("F1: 0.857 | AUC: 1.000", table_cell_style), Paragraph("F1: 1.000 | AUC: 1.000", table_cell_style)],
        [Paragraph("Ethionamide (ETH)*", table_cell_bold), Paragraph("F1: 0.000 | AUC: N/A", table_cell_style), Paragraph("F1: 0.000 | AUC: N/A", table_cell_style), Paragraph("F1: 0.000 | AUC: N/A", table_cell_style), Paragraph("F1: 0.000 | AUC: N/A", table_cell_style)],
        [Paragraph("Linezolid (LZD)*", table_cell_bold), Paragraph("F1: 1.000 | AUC: 1.000", table_cell_style), Paragraph("F1: 0.000 | AUC: 0.500", table_cell_style), Paragraph("F1: 1.000 | AUC: 1.000", table_cell_style), Paragraph("F1: 0.000 | AUC: 0.994", table_cell_style)],
        [Paragraph("<b>Macro Average (Seed 42)</b>", table_cell_bold), Paragraph("<b>F1: 0.869 | AUC: 0.998</b>", table_cell_bold), Paragraph("<b>F1: 0.657 | AUC: 0.900</b>", table_cell_bold), Paragraph("<b>F1: 0.872 | AUC: 1.000</b>", table_cell_bold), Paragraph("<b>F1: 0.707 | AUC: 0.991</b>", table_cell_bold)],
        [Paragraph("<b>5-Seed Mean &plusmn; Std (F1)</b>", table_cell_bold), Paragraph("0.8692 &plusmn; 0.0004", table_cell_bold), Paragraph("0.6571 &plusmn; 0.0011", table_cell_bold), Paragraph("<b>0.8718 &plusmn; 0.0199</b>", table_cell_bold), Paragraph("0.7396 &plusmn; 0.0212", table_cell_bold)],
        [Paragraph("<b>5-Seed Macro ROC-AUC</b>", table_cell_bold), Paragraph("0.9984 &plusmn; 0.0000", table_cell_bold), Paragraph("0.9003 &plusmn; 0.0001", table_cell_bold), Paragraph("<b>0.9999 &plusmn; 0.0000</b>", table_cell_bold), Paragraph("0.9910 &plusmn; 0.0017", table_cell_bold)],
    ]
    mc_table = Table(model_comp_data, colWidths=[1.5 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch, 1.35 * inch])
    mc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('BACKGROUND', (0, 10), (-1, 10), colors.HexColor("#edf2f7")),
        ('BACKGROUND', (0, 11), (-1, 12), colors.HexColor("#e6fffa")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.0),
    ]))
    story.append(mc_table)
    story.append(Paragraph("Table 4.1: Test set performance comparison across 9 target drugs on the clinical Afro-TB cohort (Catalogue Protocol). *Note: ETH has 0 test positives (AUC undefined), LZD has 2 test positives, and CAP has 4 test positives.", caption_style))
    story.append(PageBreak())

    # Section 4.3.2 (Page 14)
    story.append(Paragraph("4.3.2 Breakthrough Finding: Label Tautology Audit &amp; In-Silico Leave-Drug-Out (LDO) Validation", h3_style))
    story.append(Paragraph(
        "Our Review 2 audit of the raw Afro-TB dataset uncovered that <code>y_amr[i, d]</code> is an exact Boolean OR over catalogue feature columns: "
        "a zero-parameter rule reproduces the 13,753-isolate label matrix with <b>100% exact match (Macro-F1 = 1.000)</b> without any learning. "
        "Thus, high catalogue baseline scores merely reflect deterministic OR-gate memorization. Furthermore, our <code>GatedHybridGNN</code> "
        "architecture converged to a mean gating parameter of <b>&gamma; = 0.943 (100% of isolates &gt; 0.9)</b>, proving that when features are exact OR gates, "
        "deep neural networks learn to explicitly disable graph message passing to avoid signal dilution. "
        "<br/>To rigorously evaluate whether phylogenetic networks add predictive value when mutational determinants are missing or uncharacterized, "
        "we developed Leave-Drug-Out (LDO) and non-catalogue genome-wide protocols on a calibrated in-silico simulation replica (6,000 isolates "
        "mirroring Afro-TB's clonal structure and prevalence). Under LDO, <b>GCN + Skip achieves 0.5034 core-6 macro-F1, significantly outperforming "
        "graph-free MLP (0.4491, p &lt; 0.001) and Random Forest (0.3186)</b>. Under strict Clade Hold-Out splits, GCN+Skip maintains its lead "
        "(0.4754 vs MLP 0.4207 and RF 0.2504). This provides conclusive proof of our core thesis: phylogenetic graphs recover resistance by propagating "
        "transmission clade signals when canonical mutations are absent.",
        body_style
    ))

    res_img_path = ASSETS_DIR / "results_charts.png"
    if res_img_path.exists():
        story.append(Image(str(res_img_path), width=6.8 * inch, height=2.5 * inch))
        story.append(Paragraph("Figure 4.6: (A) In-silico simulation benchmark across evaluation protocols showing GNN superiority under Leave-Drug-Out; (B) Clinical Afro-TB GNN 2&times;2 factorial ablation results across 5 seeds.", caption_style))

    proto_data = [
        [Paragraph("<b>Evaluation Protocol</b>", table_header_style), Paragraph("<b>Split Strategy</b>", table_header_style), Paragraph("<b>MLP (No Graph)</b>", table_header_style), Paragraph("<b>GCN + Skip (Phylo-GNN)</b>", table_header_style), Paragraph("<b>Random Forest</b>", table_header_style), Paragraph("<b>Theoretical Ceiling</b>", table_header_style)],
        [Paragraph("Catalogue (Status Quo)", table_cell_bold), Paragraph("Original Stratified", table_cell_style), Paragraph("0.9958 &plusmn; 0.0009", table_cell_style), Paragraph("0.9436 &plusmn; 0.0017", table_cell_style), Paragraph("0.9386 &plusmn; 0.0054", table_cell_style), Paragraph("1.0000 (OR Rule)", table_cell_style)],
        [Paragraph("Catalogue (Status Quo)", table_cell_bold), Paragraph("Phylo-Clade Held-Out", table_cell_style), Paragraph("0.9846 &plusmn; 0.0067", table_cell_style), Paragraph("0.7339 &plusmn; 0.0574", table_cell_style), Paragraph("0.8757 &plusmn; 0.0034", table_cell_style), Paragraph("1.0000 (OR Rule)", table_cell_style)],
        [Paragraph("Genome-wide SNPs (1500 sites)", table_cell_bold), Paragraph("Original Stratified", table_cell_style), Paragraph("0.4801 &plusmn; 0.0033", table_cell_style), Paragraph("0.4769 &plusmn; 0.0029", table_cell_style), Paragraph("0.4763 &plusmn; 0.0032", table_cell_style), Paragraph("0.5106 (Oracle)*", table_cell_style)],
        [Paragraph("Genome-wide SNPs (1500 sites)", table_cell_bold), Paragraph("Phylo-Clade Held-Out", table_cell_style), Paragraph("0.2623 &plusmn; 0.0094", table_cell_style), Paragraph("0.2464 &plusmn; 0.0145", table_cell_style), Paragraph("0.2635 &plusmn; 0.0068", table_cell_style), Paragraph("0.2676 (Oracle)*", table_cell_style)],
        [Paragraph("<b>Leave-Drug-Out (LDO)</b>", table_cell_bold), Paragraph("<b>Original Stratified</b>", table_cell_bold), Paragraph("0.4491 &plusmn; 0.0114", table_cell_style), Paragraph("<b>0.5034 &plusmn; 0.0025</b>", table_cell_bold), Paragraph("0.3186 &plusmn; 0.0014", table_cell_style), Paragraph("—", table_cell_style)],
        [Paragraph("<b>Leave-Drug-Out (LDO)</b>", table_cell_bold), Paragraph("<b>Phylo-Clade Held-Out</b>", table_cell_bold), Paragraph("0.4207 &plusmn; 0.0053", table_cell_style), Paragraph("<b>0.4754 &plusmn; 0.0054</b>", table_cell_bold), Paragraph("0.2504 &plusmn; 0.0009", table_cell_style), Paragraph("—", table_cell_style)],
    ]
    proto_table = Table(proto_data, colWidths=[1.7 * inch, 1.35 * inch, 1.05 * inch, 1.3 * inch, 1.05 * inch, 1.3 * inch])
    proto_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
        ('BACKGROUND', (0, 5), (-1, 6), colors.HexColor("#e6fffa")),
        ('TOPPADDING', (0, 0), (-1, -1), 2.0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.0),
    ]))
    story.append(proto_table)
    story.append(Paragraph("Table 4.2: In-silico simulation benchmark on calibrated Afro-TB replica across evaluation protocols, splits, and model architectures (Core-6 Macro-F1 &plusmn; Std across 3 seeds). *Note: Oracle ceiling is calculated from the simulation generator's known Bayesian clade probabilities and exists only in the simulation fixture.", caption_style))
    story.append(PageBreak())

    # ==========================================
    # 9. SECTION 5: REFERENCES (Page 15)
    # ==========================================
    ref_heading_style = ParagraphStyle(
        'RefSubHeading',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=5,
        spaceAfter=2,
        keepWithNext=True
    )

    ref_body_style = ParagraphStyle(
        'RefBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=12,
        alignment=4,
        textColor=colors.HexColor('#2d3748'),
        spaceAfter=2.5
    )

    story.append(Paragraph("5. REFERENCES", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2b6cb0"), spaceAfter=5))

    story.append(Paragraph("Weblinks:", ref_heading_style))
    weblinks = [
        "[1] Afro-TB Project Repository &amp; Documentation, GitHub: https://github.com/Bondae1103/AfroTB-phylo-GNN-AMR-predictor",
        "[2] World Health Organization, <i>Catalogue of mutations in Mycobacterium tuberculosis complex and their association with drug resistance</i>, WHO Guidelines, Geneva, 2021. https://www.who.int/publications/i/item/9789240028173",
        "[3] PyTorch Geometric (PyG) Graph Deep Learning Library Documentation, https://pytorch-geometric.readthedocs.io/",
        "[4] The CRyPTIC Consortium, Comprehensive Resistance Prediction for Tuberculosis Data Portal, http://cryptic.ox.ac.uk/"
    ]
    for w in weblinks:
        story.append(Paragraph(w, ref_body_style))

    story.append(Paragraph("Journals: &lt;IEEE Format&gt;", ref_heading_style))
    journals = [
        "[5] M. Laamarti et al., \"Afro-TB: a curated database of Mycobacterium tuberculosis whole-genome sequences from Africa,\" <i>Scientific Data</i>, vol. 10, no. 1, p. 742, Dec. 2023, doi: 10.1038/s41597-023-02641-z.",
        "[6] F. Coll et al., \"A robust SNP barcode for typing Mycobacterium tuberculosis complex strains,\" <i>Nature Communications</i>, vol. 5, no. 1, p. 4812, Sep. 2014, doi: 10.1038/ncomms5812.",
        "[7] A. L. Manson et al., \"Genomic analysis of globally diverse Mycobacterium tuberculosis strains provides insights into the emergence and spread of multidrug resistance,\" <i>Nature Genetics</i>, vol. 49, no. 3, pp. 395–402, Mar. 2017, doi: 10.1038/ng.3767.",
        "[8] The CRyPTIC Consortium, \"A data compendium associating the genomes of 12,288 Mycobacterium tuberculosis isolates with quantitative resistance to 13 antimicrobials,\" <i>PLOS Biology</i>, vol. 20, no. 8, p. e3001721, Aug. 2022, doi: 10.1371/journal.pbio.3001721.",
        "[9] Y. Yang et al., \"Machine learning for classifying tuberculosis drug resistance from DNA sequencing data,\" <i>Bioinformatics</i>, vol. 34, no. 9, pp. 1509–1517, May 2018, doi: 10.1093/bioinformatics/btx801.",
        "[10] E. S. Kavvas et al., \"Machine learning and structural analysis of Mycobacterium tuberculosis pan-genome identifies genetic signatures of antibiotic resistance,\" <i>Nature Communications</i>, vol. 9, no. 1, p. 4306, Oct. 2018, doi: 10.1038/s41467-018-06634-y.",
        "[11] T. N. Kipf and M. Welling, \"Semi-Supervised Classification with Graph Convolutional Networks,\" in <i>International Conference on Learning Representations (ICLR)</i>, 2017."
    ]
    for j in journals:
        story.append(Paragraph(j, ref_body_style))

    story.append(Paragraph("Conference: &lt;IEEE Format&gt;", ref_heading_style))
    conferences = [
        "[12] S. Brody, U. Alon, and E. Yahav, \"How Attentive are Graph Attention Networks?,\" in <i>International Conference on Learning Representations (ICLR)</i>, 2022.",
        "[13] P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio, \"Graph Attention Networks,\" in <i>International Conference on Learning Representations (ICLR)</i>, 2018.",
        "[14] W. L. Hamilton, R. Ying, and J. Leskovec, \"Inductive representation learning on large graphs,\" in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 30, 2017, pp. 1024–1034."
    ]
    for c in conferences:
        story.append(Paragraph(c, ref_body_style))

    story.append(Paragraph("Book:", ref_heading_style))
    books = [
        "[15] I. Goodfellow, Y. Bengio, and A. Courville, <i>Deep Learning</i>, Cambridge, MA, USA: MIT Press, 2016.",
        "[16] R. Durbin, S. R. Eddy, A. Krogh, and G. Mitchison, <i>Biological Sequence Analysis: Probabilistic Models of Proteins and Nucleic Acids</i>, Cambridge, UK: Cambridge University Press, 1998.",
        "[17] World Health Organization, <i>Global Tuberculosis Report 2024</i>, Geneva: World Health Organization, 2024."
    ]
    for b in books:
        story.append(Paragraph(b, ref_body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated Review 2 Project Report: {OUTPUT_PDF}")


if __name__ == "__main__":
    build_pdf()
