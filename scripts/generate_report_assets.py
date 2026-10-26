"""Generate high-resolution figures for Review 2 Project Report.

Creates:
1. system_architecture.png - End-to-end data pipeline and GNN model architecture
2. dfd_diagram.png - Level 0 Context and Level 1 Data Flow Diagram
3. use_case_diagram.png - Use Case Diagram with actors and use cases
4. class_diagram.png - Core software class architecture diagram
5. sequence_diagram.png - Pipeline execution sequence diagram
6. gantt_chart.png - Project plan Gantt chart highlighting Review 2 (70% milestone)
7. results_charts.png - Multi-panel performance comparison and ablation results
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUTPUT_DIR = Path("report_assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'


def create_gantt_chart():
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
    
    tasks = [
        ("Phase 1: Problem Definition & Sourcing", 1, 3, 100),
        ("Phase 2: Raw VCF Streaming & QC Audits", 2, 4, 100),
        ("Phase 3: Core SNP Matrix & Jaccard Graph", 4, 6, 100),
        ("Phase 4: Tabular Baselines (RF, XGB, MLP)", 5, 7, 100),
        ("Phase 5: Multi-Task GNN & Skip Connections", 6, 8, 100),
        ("Phase 6: Label Tautology Audit & LDO", 7, 9, 100),
        ("Phase 7: Lineage Protocol & Phylo Splits", 8, 9, 100),
        ("Phase 8: Explainability (GNNExplainer)", 10, 12, 0),
        ("Phase 9: External Cross-Cohort Testing", 11, 13, 0),
        ("Phase 10: Web Dashboard & Thesis", 12, 14, 0)
    ]
    
    y_pos = np.arange(len(tasks))
    
    for i, (task, start, end, progress) in enumerate(tasks):
        duration = end - start
        color = '#2b5c8f' if progress == 100 else '#a0aec0'
        ax.barh(i, duration, left=start, height=0.55, align='center',
                color=color, edgecolor='#1a365d', alpha=0.85 if progress == 100 else 0.4,
                zorder=3)
        
        status_txt = f"{progress}% Complete" if progress == 100 else "Pending (Review 3)"
        ax.text(end + 0.15, i, status_txt, va='center', ha='left',
                fontsize=8, fontweight='bold' if progress == 100 else 'normal',
                color='#1e3a8a' if progress == 100 else '#718096')

    # Current Review 2 milestone line
    ax.axvline(x=9.0, color='#e53e3e', linestyle='--', linewidth=2, zorder=5, label='Current Milestone: Review 2 (~70% Completion)')
    ax.axvline(x=4.0, color='#3182ce', linestyle=':', linewidth=1.5, zorder=5, label='Review 1 (30% Target)')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([t[0] for t in tasks], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel('Project Timeline (Weeks)', fontsize=9.5, fontweight='bold')
    ax.set_xlim(0.5, 17)
    ax.set_xticks(range(1, 15))
    ax.set_xticklabels([f"W{w}" for w in range(1, 15)], fontsize=8.5)
    ax.grid(axis='x', linestyle='--', alpha=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.set_title('AfroTB Phylo-GNN Project Plan & Execution Timeline (Review 2: ~70% Milestone)',
                 fontsize=10.5, fontweight='bold', pad=10)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gantt_chart.png", dpi=300)
    plt.close()
    print("Saved gantt_chart.png")


def create_system_architecture():
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.axis('off')

    # Styles
    box_blue = dict(boxstyle="round,pad=0.4", fc="#ebf8ff", ec="#3182ce", lw=1.5)
    box_teal = dict(boxstyle="round,pad=0.4", fc="#e6fffa", ec="#319795", lw=1.5)
    box_purple = dict(boxstyle="round,pad=0.4", fc="#faf5ff", ec="#805ad5", lw=1.5)
    box_orange = dict(boxstyle="round,pad=0.4", fc="#fffaf0", ec="#dd6b20", lw=1.5)

    # Section 1: Raw Data & Ingestion
    rect1 = patches.FancyBboxPatch((0.4, 4.8), 2.8, 3.2, boxstyle="round,pad=0.1",
                                  fc="#f7fafc", ec="#cbd5e0", lw=1.2, linestyle='--')
    ax.add_patch(rect1)
    ax.text(1.8, 7.75, "1. Raw Genomic Ingestion\n& Strict QC Audits", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color="#2b6cb0")
    
    ax.text(1.8, 6.7, "AfroTB Dataset (N=13,753)\n• 13,753 Per-Isolate VCFs\n• 157-Mutation Catalogue\n• WHO Drug Annotations",
            ha='center', va='center', fontsize=8, bbox=box_blue)
    ax.text(1.8, 5.35, "Streaming QC Engine\n• 1:1 ID Match (13,753/13,753)\n• Core SNPs (465,257 sites)\n• Missingness Filter (<10%)",
            ha='center', va='center', fontsize=7.5, bbox=box_blue)

    # Section 2: Graph & Feature Construction
    rect2 = patches.FancyBboxPatch((3.7, 4.8), 3.2, 3.2, boxstyle="round,pad=0.1",
                                  fc="#f7fafc", ec="#cbd5e0", lw=1.2, linestyle='--')
    ax.add_patch(rect2)
    ax.text(5.3, 7.75, "2. Phylogeny Graph &\nFeature Construction", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color="#2c7a7b")

    ax.text(5.3, 6.7, "Core-SNP Sparse Matrix\nShape: (13,753 × 465,257)\nPass 1: Site Frequency Filter\nPass 2: Binary Call Extraction",
            ha='center', va='center', fontsize=7.5, bbox=box_teal)
    ax.text(5.3, 5.35, "Jaccard k-NN Graph (k=20)\n• d(i,j) = 1 - |A ∩ B| / |A ∪ B|\n• Union Mode (404,944 edges)\n• 98.3% Giant Component",
            ha='center', va='center', fontsize=7.5, bbox=box_teal)

    # Section 3: Evaluation Protocols & Tautology Resolution
    rect3 = patches.FancyBboxPatch((7.4, 4.8), 3.2, 3.2, boxstyle="round,pad=0.1",
                                  fc="#f7fafc", ec="#cbd5e0", lw=1.2, linestyle='--')
    ax.add_patch(rect3)
    ax.text(9.0, 7.75, "3. Rigorous Evaluation Protocols\n& Tautology Audit", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color="#c05621")

    ax.text(9.0, 6.7, "Label Tautology Discovery\ny_amr = exact Boolean OR(X)\n• Catalogue: OR Reconstruction\n• Zero-param Rule: F1=1.00",
            ha='center', va='center', fontsize=7.5, bbox=box_orange)
    ax.text(9.0, 5.35, "Valid Unbiased Protocols:\n• Leave-Drug-Out (LDO)\n• Genome-wide SNP (1500 sites)\n• Clade-Held-Out Phylo Splits",
            ha='center', va='center', fontsize=7.5, bbox=box_orange)

    # Section 4: Deep GNN Architecture (Bottom)
    rect4 = patches.FancyBboxPatch((0.4, 0.4), 10.2, 3.9, boxstyle="round,pad=0.1",
                                  fc="#f7fafc", ec="#cbd5e0", lw=1.2, linestyle='--')
    ax.add_patch(rect4)
    ax.text(5.5, 4.05, "4. Multi-Task Phylogeny-Aware Graph Neural Network Architecture",
            ha='center', va='center', fontsize=10.5, fontweight='bold', color="#553c9e")

    # Inside GNN architecture
    ax.text(1.6, 2.7, "Input Node Features\n• Catalogue X (157 cols)\n• Genome-wide SNPs (1500 cols)\n• LDO Masked Features",
            ha='center', va='center', fontsize=7.5, bbox=box_purple)
    
    ax.text(4.0, 2.7, "Layer 1: GCNConv / GATv2\n(In_Dim → 64 Channels)\n• Jaccard Edge Weights\n• Dynamic Edge Attention\n• ReLU + Dropout (0.3)",
            ha='center', va='center', fontsize=7.5, bbox=box_purple)
    
    ax.text(6.6, 2.7, "Layer 2: GCNConv / GATv2\n(64 → 64 Channels)\n• Message Aggregation\n• High-Order Clonal Context\n• Graph Neighborhood Mix",
            ha='center', va='center', fontsize=7.5, bbox=box_purple)

    ax.text(9.2, 2.7, "Skip Connection Block\nConcatenates Raw Input:\nh = [h_GNN || X_input]\n(64 + In_Dim Channels)\nPrevents Oversmoothing",
            ha='center', va='center', fontsize=7.5, bbox=box_purple)

    # Output Heads
    ax.text(3.5, 1.2, "Head 1: Multi-Label AMR Prediction\n9 Sigmoid Outputs: RIF, INH, EMB, PZA, STM, LEV, CAP, ETH, LZD\nLoss: Weighted Binary Cross-Entropy (Class-Balanced)",
            ha='center', va='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.4", fc="#feebc8", ec="#c05621", lw=1.5))

    ax.text(8.0, 1.2, "Head 2: Multi-Class Lineage Prediction\nLogits over M.tb Lineages (BOV_AFRI, L1-L4)\nLoss: CrossEntropyLoss (ignore_index=-100)",
            ha='center', va='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.4", fc="#e9d8fd", ec="#6b46c1", lw=1.5))

    # Connectors / Arrows
    arrow_props = dict(facecolor='#4a5568', edgecolor='#2d3748', width=1.5, headwidth=6, shrink=0.05)
    ax.annotate('', xy=(3.7, 6.7), xytext=(3.2, 6.7), arrowprops=arrow_props)
    ax.annotate('', xy=(7.4, 6.7), xytext=(6.9, 6.7), arrowprops=arrow_props)
    ax.annotate('', xy=(1.6, 3.4), xytext=(1.8, 4.8), arrowprops=arrow_props)
    ax.annotate('', xy=(4.0, 3.4), xytext=(5.3, 4.8), arrowprops=arrow_props)
    ax.annotate('', xy=(2.7, 2.7), xytext=(2.4, 2.7), arrowprops=arrow_props)
    ax.annotate('', xy=(5.3, 2.7), xytext=(5.0, 2.7), arrowprops=arrow_props)
    ax.annotate('', xy=(7.9, 2.7), xytext=(7.6, 2.7), arrowprops=arrow_props)
    ax.annotate('', xy=(3.5, 1.9), xytext=(5.5, 2.1), arrowprops=arrow_props)
    ax.annotate('', xy=(8.0, 1.9), xytext=(6.5, 2.1), arrowprops=arrow_props)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "system_architecture.png", dpi=300)
    plt.close()
    print("Saved system_architecture.png")


def create_dfd_diagram():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 8.5), dpi=300, gridspec_kw={'height_ratios': [1, 2.2]})
    
    # Subplot 1: Level 0 DFD (Context Diagram)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 3)
    ax1.axis('off')
    ax1.set_title("Data Flow Diagram — Level 0 (Context Level)", fontsize=10.5, fontweight='bold', pad=8)
    
    ent_style = dict(boxstyle="square,pad=0.5", fc="#e2e8f0", ec="#4a5568", lw=1.5)
    proc_style = dict(boxstyle="circle,pad=0.5", fc="#ebf8ff", ec="#3182ce", lw=2)
    
    ax1.text(1.2, 1.5, "Bioinformatics\nResearcher /\nMicrobiologist", ha='center', va='center', fontsize=8, bbox=ent_style)
    ax1.text(5.0, 1.5, "0.0\nAfroTB Phylo-GNN\nPrediction\nSystem", ha='center', va='center', fontsize=8.5, fontweight='bold', bbox=proc_style)
    ax1.text(8.8, 1.5, "Public Health /\nClinical Decision\nSupport", ha='center', va='center', fontsize=8, bbox=ent_style)
    
    arr = dict(facecolor='#2d3748', edgecolor='#2d3748', width=1.2, headwidth=5)
    ax1.annotate('Raw VCFs / Mutations', xy=(3.7, 1.7), xytext=(2.3, 1.7), arrowprops=arr, fontsize=7.5, va='bottom', ha='center')
    ax1.annotate('AMR & Lineage Queries', xy=(3.7, 1.2), xytext=(2.3, 1.2), arrowprops=arr, fontsize=7.5, va='top', ha='center')
    ax1.annotate('9-Drug AMR Probabilities', xy=(7.7, 1.7), xytext=(6.3, 1.7), arrowprops=arr, fontsize=7.5, va='bottom', ha='center')
    ax1.annotate('Lineage & Co-Resistances', xy=(7.7, 1.2), xytext=(6.3, 1.2), arrowprops=arr, fontsize=7.5, va='top', ha='center')

    # Subplot 2: Level 1 DFD (Decomposition)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 7)
    ax2.axis('off')
    ax2.set_title("Data Flow Diagram — Level 1 (Functional Decomposition)", fontsize=10.5, fontweight='bold', pad=8)
    
    p_box = dict(boxstyle="round,pad=0.35", fc="#ebf8ff", ec="#3182ce", lw=1.5)
    d_box = dict(boxstyle="sawtooth,pad=0.3", fc="#fefcbf", ec="#d69e2e", lw=1.2)
    e_box = dict(boxstyle="square,pad=0.4", fc="#e2e8f0", ec="#4a5568", lw=1.5)

    # Entities & Data Stores
    ax2.text(0.8, 5.5, "Entity:\nAfro-TB Data\nRepository", ha='center', va='center', fontsize=7.5, bbox=e_box)
    ax2.text(0.8, 2.0, "Entity:\nResearcher /\nClinician", ha='center', va='center', fontsize=7.5, bbox=e_box)
    
    # Processes
    ax2.text(3.0, 5.5, "1.0\nIngest & QC\nStreaming VCFs", ha='center', va='center', fontsize=7.5, fontweight='bold', bbox=p_box)
    ax2.text(6.0, 5.5, "2.0\nConstruct Core\nSNP Jaccard Graph", ha='center', va='center', fontsize=7.5, fontweight='bold', bbox=p_box)
    ax2.text(9.0, 5.5, "3.0\nPartition & LDO\nFeature Protocol", ha='center', va='center', fontsize=7.5, fontweight='bold', bbox=p_box)
    
    ax2.text(6.0, 3.2, "4.0\nTrain Multi-Task\nSkip GNN Model", ha='center', va='center', fontsize=7.5, fontweight='bold', bbox=p_box)
    ax2.text(6.0, 1.0, "5.0\nMulti-Metric\nEvaluation & Inference", ha='center', va='center', fontsize=7.5, fontweight='bold', bbox=p_box)

    # Data Stores
    ax2.text(4.5, 4.3, "D1: Processed Matrices & Graph (pyg_graph.pt)", ha='center', va='center', fontsize=7.2, bbox=d_box)
    ax2.text(8.0, 2.1, "D2: Trained GNN & Baseline Weights", ha='center', va='center', fontsize=7.2, bbox=d_box)
    ax2.text(3.0, 2.1, "D3: Benchmark Metrics & QC Reports", ha='center', va='center', fontsize=7.2, bbox=d_box)

    # Flow arrows
    ax2.annotate('', xy=(2.0, 5.5), xytext=(1.5, 5.5), arrowprops=arr)
    ax2.annotate('Raw VCFs', xy=(2.0, 5.7), xytext=(1.5, 5.7), fontsize=6.8, va='bottom')
    
    ax2.annotate('', xy=(5.0, 5.5), xytext=(4.0, 5.5), arrowprops=arr)
    ax2.annotate('Cleaned Calls', xy=(4.5, 5.7), xytext=(4.5, 5.7), fontsize=6.8, va='bottom', ha='center')

    ax2.annotate('', xy=(8.0, 5.5), xytext=(7.0, 5.5), arrowprops=arr)
    ax2.annotate('k-NN Graph', xy=(7.5, 5.7), xytext=(7.5, 5.7), fontsize=6.8, va='bottom', ha='center')

    ax2.annotate('', xy=(4.5, 4.7), xytext=(5.5, 5.1), arrowprops=arr)
    ax2.annotate('', xy=(5.5, 3.8), xytext=(4.5, 4.0), arrowprops=arr)

    ax2.annotate('', xy=(6.0, 2.6), xytext=(6.0, 1.6), arrowprops=arr)
    ax2.annotate('Model Logits', xy=(6.6, 2.1), xytext=(6.6, 2.1), fontsize=6.8, va='center')

    ax2.annotate('', xy=(7.0, 3.2), xytext=(7.5, 2.5), arrowprops=arr)
    ax2.annotate('', xy=(5.0, 1.0), xytext=(3.8, 1.8), arrowprops=arr)

    ax2.annotate('', xy=(1.6, 2.0), xytext=(5.0, 1.0), arrowprops=arr)
    ax2.annotate('AMR & Lineage Reports', xy=(2.8, 1.4), xytext=(2.8, 1.4), fontsize=6.8, va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dfd_diagram.png", dpi=300)
    plt.close()
    print("Saved dfd_diagram.png")


def create_use_case_diagram():
    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title("Use Case Diagram — AfroTB Phylogeny-Aware GNN System", fontsize=11, fontweight='bold', pad=10)

    # System boundary
    rect = patches.FancyBboxPatch((2.2, 0.4), 5.6, 7.2, boxstyle="round,pad=0.1",
                                  fc="#ffffff", ec="#2b6cb0", lw=2, linestyle='-')
    ax.add_patch(rect)
    ax.text(5.0, 7.35, "AfroTB Prediction System Boundary", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color="#2b6cb0")

    # Actors
    def draw_actor(x, y, label):
        circle = patches.Circle((x, y + 0.4), 0.22, fc="#ffffff", ec="#2d3748", lw=1.8)
        ax.add_patch(circle)
        ax.plot([x, x], [y + 0.18, y - 0.25], color="#2d3748", lw=1.8)
        ax.plot([x - 0.25, x + 0.25], [y + 0.05, y + 0.05], color="#2d3748", lw=1.8)
        ax.plot([x, x - 0.2], [y - 0.25, y - 0.6], color="#2d3748", lw=1.8)
        ax.plot([x, x + 0.2], [y - 0.25, y - 0.6], color="#2d3748", lw=1.8)
        ax.text(x, y - 0.85, label, ha='center', va='top', fontsize=8, fontweight='bold', color="#1a202c")

    draw_actor(1.0, 5.8, "Bioinformatics\nResearcher")
    draw_actor(1.0, 2.2, "Clinical\nMicrobiologist")
    draw_actor(9.0, 4.0, "Automated CI/QC\nPipeline")

    use_cases = [
        (5.0, 6.6, "UC1: Stream & Validate\n13,753 Raw VCF Files"),
        (5.0, 5.5, "UC2: Compute Jaccard\nPhylogenetic k-NN Graph"),
        (5.0, 4.4, "UC3: Execute Label\nTautology Audit"),
        (5.0, 3.3, "UC4: Train Multi-Task\nSkip GNN Models"),
        (5.0, 2.2, "UC5: Predict 9-Drug AMR\nunder LDO Protocols"),
        (5.0, 1.1, "UC6: Jointly Predict M.tb\nLineage & Co-Resistance"),
    ]

    for x, y, text in use_cases:
        ellipse = patches.Ellipse((x, y), 3.4, 0.8, fc="#f0fff4", ec="#38a169", lw=1.4)
        ax.add_patch(ellipse)
        ax.text(x, y, text, ha='center', va='center', fontsize=7.5, fontweight='bold', color="#22543d")

    line_kw = dict(color="#4a5568", lw=1.2)
    ax.plot([1.3, 3.3], [5.8, 6.6], **line_kw)
    ax.plot([1.3, 3.3], [5.8, 5.5], **line_kw)
    ax.plot([1.3, 3.3], [5.8, 4.4], **line_kw)
    ax.plot([1.3, 3.3], [5.8, 3.3], **line_kw)

    ax.plot([1.3, 3.3], [2.2, 3.3], **line_kw)
    ax.plot([1.3, 3.3], [2.2, 2.2], **line_kw)
    ax.plot([1.3, 3.3], [2.2, 1.1], **line_kw)

    ax.plot([8.7, 6.7], [4.0, 6.6], **line_kw)
    ax.plot([8.7, 6.7], [4.0, 5.5], **line_kw)
    ax.plot([8.7, 6.7], [4.0, 4.4], **line_kw)

    ax.annotate('<<include>>', xy=(5.0, 5.1), xytext=(5.0, 4.8), fontsize=6.5, ha='center', color='#718096', fontstyle='italic')
    ax.annotate('<<include>>', xy=(5.0, 4.0), xytext=(5.0, 3.7), fontsize=6.5, ha='center', color='#718096', fontstyle='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "use_case_diagram.png", dpi=300)
    plt.close()
    print("Saved use_case_diagram.png")


def create_class_diagram():
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    ax.set_title("Class Diagram — Software Architecture & Object Hierarchy", fontsize=11, fontweight='bold', pad=10)

    def draw_class(x, y, w, h, name, attrs, methods, header_bg="#2b6cb0"):
        h_rect = patches.Rectangle((x, y + h - 0.45), w, 0.45, fc=header_bg, ec="#1a365d", lw=1.2)
        ax.add_patch(h_rect)
        ax.text(x + w / 2, y + h - 0.22, name, ha='center', va='center', fontsize=8, fontweight='bold', color="#ffffff")
        
        b_rect = patches.Rectangle((x, y), w, h - 0.45, fc="#ffffff", ec="#1a365d", lw=1.2)
        ax.add_patch(b_rect)
        
        curr_y = y + h - 0.65
        for attr in attrs:
            ax.text(x + 0.1, curr_y, f"- {attr}", ha='left', va='center', fontsize=6.8, color="#2d3748")
            curr_y -= 0.22
            
        ax.plot([x, x + w], [curr_y + 0.08, curr_y + 0.08], color="#cbd5e0", lw=0.8)
        curr_y -= 0.12
        
        for method in methods:
            ax.text(x + 0.1, curr_y, f"+ {method}", ha='left', va='center', fontsize=6.8, color="#2b6cb0")
            curr_y -= 0.22

    draw_class(0.4, 5.0, 3.2, 3.0, "AfroTBDataLoader",
               ["raw_vcf_dir: Path", "features_path: Path", "labels_path: Path", "splits_path: Path"],
               ["stream_vcf_calls(): Generator", "build_sample_ids(): DataFrame", "get_split_masks(): Tuple[Tensor]"],
               header_bg="#2b6cb0")

    draw_class(4.0, 5.0, 3.4, 3.0, "PhylogeneticGraphBuilder",
               ["n_isolates: int", "core_snp_sites: int", "k_neighbors: int = 20", "metric: str = 'jaccard'"],
               ["build_snp_matrix(): sp.csr_matrix", "compute_jaccard_distance(): ndarray", "construct_knn_graph(): PyG.Data", "validate_graph_topology(): Dict"],
               header_bg="#2c7a7b")

    draw_class(7.8, 5.0, 2.8, 3.0, "BaseGNN (nn.Module)",
               ["in_dim: int", "hidden_dim: int", "n_out: int = 9", "dropout: float = 0.3"],
               ["forward(x, edge_index, w): Tensor", "embed(x, edge_index, w): Tensor", "compute_loss(preds, targets): Tensor"],
               header_bg="#553c9e")

    draw_class(0.4, 1.0, 3.2, 3.2, "MultiTaskGCN",
               ["conv1: GCNConv", "conv2: GCNConv", "skip_connection: bool = True", "amr_head: Linear(dim, 9)", "lineage_head: Linear(dim, n_lin)"],
               ["forward(x, edge_index, w): Tensor", "embed(x, edge_index, w): Tensor", "get_amr_logits(): Tensor", "get_lineage_logits(): Tensor"],
               header_bg="#6b46c1")

    draw_class(4.0, 1.0, 3.4, 3.2, "GatedHybridGNN",
               ["mlp_stream: Sequential", "gcn_stream: Sequential", "gate_layer: Linear(in_dim, 1)", "last_gate: Tensor (N, 1)"],
               ["forward(x, edge_index, w): Tensor", "embed(x, edge_index, w): Tensor", "gate_stats(): Dict[str, float]"],
               header_bg="#6b46c1")

    draw_class(7.8, 1.0, 2.8, 3.2, "BenchmarkEvaluator",
               ["protocols: List[str]", "splits: List[str]", "seeds: List[int] = [42, 43, 44]"],
               ["run_protocol_benchmark(): Dict", "tune_per_drug_thresholds(): Dict", "evaluate_core6_macro_f1(): float", "generate_qc_report(): Path"],
               header_bg="#c05621")

    arr = dict(facecolor='#4a5568', edgecolor='#2d3748', width=1.0, headwidth=4)
    ax.annotate('', xy=(2.0, 4.2), xytext=(7.8, 5.0), arrowprops=arr)
    ax.annotate('', xy=(5.7, 4.2), xytext=(8.8, 5.0), arrowprops=arr)
    ax.annotate('', xy=(4.0, 6.5), xytext=(3.6, 6.5), arrowprops=arr)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_diagram.png", dpi=300)
    plt.close()
    print("Saved class_diagram.png")


def create_sequence_diagram():
    fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.axis('off')
    ax.set_title("Sequence Diagram — End-to-End Training, Protocol Audit & Inference Workflow", fontsize=10.5, fontweight='bold', pad=10)

    lifelines = [
        (1.0, "CLI / Experiment\nRunner"),
        (2.8, "AfroTB\nDataLoader"),
        (4.8, "PhyloGraph\nEngine"),
        (6.8, "MultiTask\nGNN Model"),
        (8.8, "Benchmark\nEvaluator")
    ]

    for x, name in lifelines:
        rect = patches.FancyBboxPatch((x - 0.75, 7.5), 1.5, 0.7, boxstyle="round,pad=0.1", fc="#ebf8ff", ec="#3182ce", lw=1.5)
        ax.add_patch(rect)
        ax.text(x, 7.85, name, ha='center', va='center', fontsize=7.5, fontweight='bold', color="#2b6cb0")
        ax.plot([x, x], [7.5, 0.5], color="#a0aec0", linestyle="--", lw=1.2, zorder=1)

    calls = [
        (7.2, 1.0, 2.8, "1. Ingest VCFs & catalogue xlsx", True),
        (6.7, 2.8, 2.8, "2. Parse calls & audit 1:1 IDs", False),
        (6.2, 2.8, 4.8, "3. Pass core SNPs (465,257 sites)", True),
        (5.7, 4.8, 4.8, "4. Compute Jaccard k-NN graph (k=20)", False),
        (5.2, 4.8, 1.0, "5. Return pyg_graph.pt & edge weights", True),
        (4.7, 1.0, 8.8, "6. Run Label Tautology Audit", True),
        (4.2, 8.8, 1.0, "7. Confirm OR gate; Select LDO protocol", True),
        (3.7, 1.0, 6.8, "8. Initialize GNN (GCN+Skip / Hybrid)", True),
        (3.2, 6.8, 6.8, "9. Forward pass (x, edge_index, edge_w)", False),
        (2.7, 6.8, 6.8, "10. Backprop BCE(AMR) + CE(Lineage)", False),
        (2.2, 6.8, 8.8, "11. Forward val/test logits", True),
        (1.7, 8.8, 8.8, "12. Tune per-drug F1 thresholds on val", False),
        (1.2, 8.8, 1.0, "13. Return Core-6 / All-9 Macro-F1 & AUC", True),
    ]

    for y, x_from, x_to, msg, is_arrow in calls:
        if is_arrow:
            dx = x_to - x_from
            ax.annotate('', xy=(x_to, y), xytext=(x_from, y),
                        arrowprops=dict(arrowstyle="->", color="#2d3748", lw=1.3))
            ax.text(x_from + dx / 2, y + 0.12, msg, ha='center', va='bottom', fontsize=7.2, color="#1a202c", fontweight='bold')
        else:
            ax.plot([x_from, x_from + 0.5, x_from + 0.5, x_from],
                    [y + 0.1, y + 0.1, y - 0.15, y - 0.15], color="#2b6cb0", lw=1.2)
            ax.annotate('', xy=(x_from, y - 0.15), xytext=(x_from + 0.1, y - 0.15),
                        arrowprops=dict(arrowstyle="->", color="#2b6cb0", lw=1.2))
            ax.text(x_from + 0.6, y - 0.02, msg, ha='left', va='center', fontsize=7.0, color="#2b6cb0")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sequence_diagram.png", dpi=300)
    plt.close()
    print("Saved sequence_diagram.png")


def create_results_charts():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    # Panel 1: Protocol Comparison on In-Silico Simulation Replica
    protocols = ['Catalogue\n(Tautological)', 'Genome-wide SNPs\n(Non-Catalogue)', 'Leave-Drug-Out\n(Target Masked)']
    x = np.arange(len(protocols))
    width = 0.25

    mlp_scores = [0.9958, 0.4801, 0.4491]
    gnn_scores = [0.9436, 0.4769, 0.5034]
    rf_scores = [0.9386, 0.4763, 0.3186]

    rects1 = ax1.bar(x - width, mlp_scores, width, label='MLP (Graph-Free Control)', color='#4299e1', edgecolor='#2b6cb0')
    rects2 = ax1.bar(x, gnn_scores, width, label='GCN + Skip (Phylo-GNN)', color='#48bb78', edgecolor='#2f855a')
    rects3 = ax1.bar(x + width, rf_scores, width, label='Random Forest', color='#ed8936', edgecolor='#c05621')

    for rect in rects1 + rects2 + rects3:
        height = rect.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.2, fontweight='bold')

    ax1.set_ylabel('Test Core-6 Macro-F1 Score', fontsize=9.5, fontweight='bold')
    ax1.set_title('A. In-Silico Simulation Proof-of-Concept\n(Protocol Benchmark on Calibrated Replica)', fontsize=9.5, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(protocols, fontsize=8.5)
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right', fontsize=8, frameon=True)

    # Panel 2: Clinical Afro-TB GNN 2x2 Factorial Ablation across 5 Seeds
    ablations = [
        'Config D\n(Plain, Unw)',
        'Config C\n(+ Jaccard)',
        'Config A\n(+ Skip & Wt)',
        'Config B\n(+ Skip, Unw)'
    ]
    f1_default = [0.600, 0.611, 0.686, 0.691]
    f1_tuned = [0.628, 0.643, 0.740, 0.753]
    roc_auc = [0.960, 0.963, 0.991, 0.992]

    x2 = np.arange(len(ablations))
    w2 = 0.26

    r1 = ax2.bar(x2 - w2, f1_default, w2, label='5-Seed Macro-F1 (0.5)', color='#cbd5e0', edgecolor='#718096')
    r2 = ax2.bar(x2, f1_tuned, w2, label='5-Seed Macro-F1 (Tuned)', color='#38b2ac', edgecolor='#234e52')
    r3 = ax2.bar(x2 + w2, roc_auc, w2, label='5-Seed Macro ROC-AUC', color='#9f7aea', edgecolor='#553c9e')

    for rect in r1 + r2 + r3:
        height = rect.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7.2, fontweight='bold')

    ax2.set_ylabel('Score', fontsize=9.5, fontweight='bold')
    ax2.set_title('B. Clinical Afro-TB GNN 2×2 Factorial Ablation (5 Seeds)\n(Edge Weighting × Residual Skip Connections)', fontsize=9.5, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(ablations, fontsize=8.5)
    ax2.set_ylim(0, 1.15)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.legend(loc='lower right', fontsize=8, frameon=True)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "results_charts.png", dpi=300)
    plt.close()
    print("Saved results_charts.png")


if __name__ == "__main__":
    create_gantt_chart()
    create_system_architecture()
    create_dfd_diagram()
    create_use_case_diagram()
    create_class_diagram()
    create_sequence_diagram()
    create_results_charts()
    print("All figures successfully updated in report_assets/")
