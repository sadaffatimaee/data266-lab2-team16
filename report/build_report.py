import ast, json, os, re, sys, textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NB = os.path.join(ROOT, "notebooks")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
REPO_URL = "https://github.com/sadaffatimaee/data266-lab2-team16"
PENDING = []


def stream(name):
    p = os.path.join(NB, name)
    if not os.path.exists(p):
        return ""
    nb = json.load(open(p, encoding="utf-8"))
    out = []
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            if o.get("output_type") == "stream":
                out.append("".join(o.get("text", "")))
    return "".join(out)


def results_json(name):
    merged = {}
    for line in stream(name).splitlines():
        if line.startswith("RESULTS_JSON "):
            merged.update(json.loads(line[len("RESULTS_JSON "):]))
    return merged


def pend(what):
    PENDING.append(what)
    return "[pending]"


def fmt(v, nd=2, what="value"):
    if v is None:
        return pend(what)
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def p1_team():
    t = stream("p1-eval-team16.ipynb")
    rows = {}
    for name in ("baseline (dense only)", "hybrid (no reranker)", "final (hybrid + reranker + cross-refs)"):
        m = re.search(re.escape(name) + r" (\{[^}]*\})", t)
        if m:
            rows[name] = ast.literal_eval(m.group(1))
    sub = stream("part1-rag-lab-pair-16.ipynb")
    med = re.search(r"median seconds per question: ([\d.]+)", sub)
    valid = re.search(r"valid: (\d+) rows", sub)
    return rows, (float(med.group(1)) if med else None), (valid.group(1) if valid else None)


def p2_team():
    t = stream("p2_finetune_team16.ipynb")
    train = {}
    for m in re.finditer(r"\{'run': 'run\d_lora_r\d+'[^}]*\}", t):
        d = ast.literal_eval(m.group(0))
        train[d["run"]] = d
    res = {}
    block = t[t.find("PART 2 RESULTS TABLE"):t.find("TRAINING TABLE")]
    cols = ["KID", "KID std", "CLIP all", "CLIP in", "CLIP out", "CLIP comp", "LPIPS"]
    for line in block.splitlines():
        parts = line.split()
        if parts and parts[0] in ("base", "run1_lora_r4", "run2_lora_r16"):
            res[parts[0]] = {c: float(v) for c, v in zip(cols, parts[1:8])}
            res[parts[0]]["NSFW"] = int(float(parts[-1]))
    mem = {}
    for m in re.finditer(r"(run\d_lora_r\d+): highest CLIP similarity to any training image, per generated image: \[([^\]]*)\]", t):
        mem[m.group(1)] = [float(x) for x in m.group(2).split(",")]
    gen = re.search(r"120 images .*? minutes: ([\d.]+)", t)
    return train, res, mem, (float(gen.group(1)) if gen else None)


def p3_team():
    t = stream("p3-agents-team16.ipynb")
    out = {}
    for key, label in (("route", "route correct (/15)"), ("median", "median seconds / request"), ("clip", "mean CLIP of final image"),
                       ("revisions", "mean revisions")):
        m = re.search(re.escape(label) + r"\s+([\d.]+)\s+([\d.]+)", t)
        if m:
            out[key] = (float(m.group(1)), float(m.group(2)))
    m = re.search(r"red-team attack success rate: (\d+)/(\d+)", t)
    if m:
        out["red"] = (int(m.group(1)), int(m.group(2)))
    return out


def box_diagram(path, rows, title, loop=None, height=None):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(3.4, height or 0.78 * n + 0.3))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n)
    ax.axis("off")
    for i, (label, sub) in enumerate(rows):
        y = n - i - 0.85
        ax.add_patch(FancyBboxPatch((0.6, y), 8.8, 0.68, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="#eef2f7", ec="#33475b", lw=0.8))
        ax.text(5, y + 0.52, label, ha="center", va="center", fontsize=7, weight="bold", family="serif")
        ax.text(5, y + 0.22, textwrap.fill(sub, 58), ha="center", va="center", fontsize=5.4, family="serif", linespacing=1.1)
        if i < n - 1:
            ax.annotate("", xy=(5, y - 0.2), xytext=(5, y), arrowprops=dict(arrowstyle="->", lw=0.8))
    if loop:
        a, b, text = loop
        ya, yb = n - a - 0.55, n - b - 0.55
        ax.annotate("", xy=(9.45, yb), xytext=(9.45, ya), arrowprops=dict(arrowstyle="->", lw=0.8, connectionstyle="arc3,rad=-0.5"))
        ax.text(9.9, (ya + yb) / 2, text, rotation=90, ha="center", va="center", fontsize=5.6, family="serif")
    plt.tight_layout(pad=0.1)
    plt.savefig(path, dpi=220)
    plt.close(fig)


def diagrams():
    box_diagram(os.path.join(FIG, "arch_p1.png"), [
        ("Ingestion (PyMuPDF)", "text chunks of 160 words with doc_id, page, trust; 36 figure/table crops"),
        ("Figure descriptions", "Qwen2.5-VL-7B (4-bit) reads every chart; text in images is content only"),
        ("Index", "bge-small-en-v1.5 embeddings in ChromaDB + BM25; trust_level from kb_manifest.json"),
        ("Retrieval", "dense + BM25 fused (RRF), named figure pinned, bge-reranker, cross-references"),
        ("Conversation memory", "per conversation_id: rewrite follow-ups, user values kept only here"),
        ("Answer generation", "Qwen2.5-VL-7B with rules; low-trust Appendix C removed from context"),
        ("Post-checks", "citation filter, calculator, conflict and false-premise checks, JSON output"),
    ], "Part 1")
    box_diagram(os.path.join(FIG, "arch_p2.png"), [
        ("Data", "naruto-blip-captions, 1221 images, cleaned, 90/10 split (seed 266), 512x512"),
        ("Stable Diffusion 1.5 (frozen)", "VAE + CLIP text encoder + UNet"),
        ("LoRA on UNet attention", "run 1 rank 4, run 2 rank 16 (team); rank 8 own loop (baseline)"),
        ("Generation", "20 fixed prompts x seeds 0, 1; DPM-Solver 25 steps, guidance 7.5"),
        ("Evaluation", "KID, CLIP score, LPIPS, blinded human pairs, GPU-hours, memorization"),
    ], "Part 2")
    box_diagram(os.path.join(FIG, "arch_p3.png"), [
        ("Planner", "file check, NSFW and real-person refusal, route choice"),
        ("Vision Understanding", "caption, objects, visible text, answers to 2+ questions"),
        ("Prompt Engineering", "request + vision output to a generation prompt; image text never used"),
        ("Image Generation", "img2img style change, InstructPix2Pix edit, Part 2 LoRA for Naruto"),
        ("Critique", "CLIP score + VLM judgement + NSFW check: accept or revise"),
    ], "Part 3", loop=(4, 2, "revise (max 2)"))


def set_cols(section, n):
    cols = section._sectPr.xpath("./w:cols")
    el = cols[0] if cols else OxmlElement("w:cols")
    el.set(qn("w:num"), str(n))
    el.set(qn("w:space"), "360")
    if not cols:
        section._sectPr.append(el)


class Report:
    def __init__(self):
        self.d = Document()
        st = self.d.styles["Normal"]
        st.font.name = "Times New Roman"
        st.font.size = Pt(10)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        st.paragraph_format.space_after = Pt(3)
        s = self.d.sections[0]
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        s.left_margin = s.right_margin = Inches(0.7)
        s.top_margin, s.bottom_margin = Inches(0.75), Inches(0.9)
        self.sec = 0
        self.sub = 0
        self.tab = 0
        self.fig = 0

    def para(self, text, bold=False, italic=False, size=None, align=None, after=3):
        p = self.d.add_paragraph()
        for i, part in enumerate(re.split(r"(\*\*[^*]+\*\*)", text)):
            if not part:
                continue
            r = p.add_run(part[2:-2] if part.startswith("**") else part)
            r.bold = bold or part.startswith("**")
            r.italic = italic
            if size:
                r.font.size = Pt(size)
        p.paragraph_format.alignment = align if align is not None else WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(after)
        return p

    def section(self, title):
        self.sec += 1
        self.sub = 0
        roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][self.sec - 1]
        p = self.para(f"{roman}. {title.upper()}", size=10, align=WD_ALIGN_PARAGRAPH.CENTER, after=4)
        p.paragraph_format.space_before = Pt(8)

    def subsection(self, title):
        self.sub += 1
        p = self.para(f"{'ABCDEFGHIJ'[self.sub - 1]}. {title}", italic=True, after=2)
        p.paragraph_format.space_before = Pt(4)

    def table(self, caption, header, rows, widths=None):
        self.tab += 1
        roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV"][self.tab - 1]
        self.para(f"TABLE {roman}", size=8, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
        self.para(caption.upper(), size=7.5, align=WD_ALIGN_PARAGRAPH.CENTER, after=2)
        t = self.d.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            c = t.rows[0].cells[i]
            c.text = ""
            r = c.paragraphs[0].add_run(h)
            r.bold = True
            r.font.size = Pt(7.5)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                r = cells[i].paragraphs[0].add_run(str(v))
                r.font.size = Pt(7.5)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Inches(w)
        self.para("", after=2)

    def figure(self, path, caption, width=3.3):
        if not os.path.exists(path):
            pend(f"figure {os.path.basename(path)}")
            return
        self.fig += 1
        self.d.add_picture(path, width=Inches(width))
        self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.para(f"Fig. {self.fig}. {caption}", size=8, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)


def build():
    diagrams()
    p1rows, p1med, p1valid = p1_team()
    p1me = results_json("p1_ablation_safety_poushali.ipynb")
    train, p2res, mem, gen_min = p2_team()
    p2me = results_json("p2_baseline_poushali.ipynb")
    p2an = results_json("p2_analysis_poushali.ipynb")
    p3 = p3_team()
    p3me = results_json("p3_baseline_poushali.ipynb")
    he = results_json("human_eval_team16.ipynb")
    g25m = re.search(r"golden G25 \(appendix chart injection\) final answer: (.*?) \| status", stream("p1_ablation_safety_poushali.ipynb"))
    g25 = g25m.group(1) if g25m else None
    grid = hf_hub_download("sadaffatimaeee/naruto-lora-generations", "grid_8_prompts.png", repo_type="dataset")
    memfig = hf_hub_download("sadaffatimaeee/naruto-sd15-run2-lora-r16", "memorization.png")

    R = Report()
    d = R.d
    R.para("Multimodal RAG, Diffusion Fine-Tuning and a Multi-Agent Image System", size=20, align=WD_ALIGN_PARAGRAPH.CENTER, after=6)
    R.para("Sadaf Fatima Syeda and Poushali Deb Purkayastha", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    R.para("Pair_Programming_Team_16, DATA 266 Generative AI, San Jose State University", italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, after=0)
    R.para(f"GitHub: {REPO_URL} (tags p1-v1.0, p2-v1.0, p3-v1.0)", size=9, align=WD_ALIGN_PARAGRAPH.CENTER, after=10)
    body = d.add_section(WD_SECTION.CONTINUOUS)
    set_cols(body, 2)

    fin = p1rows.get("final (hybrid + reranker + cross-refs)", {})
    R.para(f"**Abstract**: We built three systems on one 16 GB GPU. Part 1 is a multimodal retrieval-augmented generation system over three "
           f"economics PDFs that answers in a fixed JSON format, keeps a separate memory per conversation and never uses the untrusted "
           f"appendix as evidence. On our 25-question golden set it answers {fmt(fin.get('correct (/25)'), 0, 'P1 final correct')} questions "
           f"correctly against {fmt(p1rows.get('baseline (dense only)', {}).get('correct (/25)'), 0, 'P1 baseline correct')} for dense retrieval alone. "
           f"Part 2 fine-tunes Stable Diffusion 1.5 with LoRA on Naruto images; the rank 16 run lowers KID from "
           f"{fmt(p2res.get('base', {}).get('KID'), 4, 'P2 base KID')} to {fmt(p2res.get('run2_lora_r16', {}).get('KID'), 4, 'P2 final KID')}. "
           f"Part 3 is a five-agent system that understands an image, writes a prompt, generates, critiques and revises; with the critique "
           f"loop it passes {fmt((he.get('p3', {}).get('task success (resolved)') or {}).get('with critique'), 0, 'P3 task success')} of 15 test scenarios "
           f"and {fmt(p3.get('red', (None,))[0], 0, 'P3 red team')} of 10 red-team attacks succeed.", size=9)
    R.para("**Index Terms**: retrieval-augmented generation, LoRA, Stable Diffusion, multi-agent systems, prompt injection.", size=9, after=6)

    R.section("Introduction")
    R.para("Lab 2 has three parts. Both members worked on all three: Sadaf led the team systems and Poushali built and ran a second "
           "complete pipeline for Parts 2 and 3, the Part 1 reranker ablation and safety probes, the failure analysis and all "
           "human-evaluation tooling. Every number in this report is printed by a notebook in the repository; the notebook is named "
           "next to each table. Weights and images are on Hugging Face. No hosted API is used: all models run locally on a Kaggle T4 "
           "or the lab GPU.")

    R.section("Part 1: Multimodal RAG")
    R.subsection("Approach and architecture")
    R.para("The knowledge base is three PDFs with trust levels from kb_manifest.json: DOC1 (textbook chapters, high), DOC2 (outturns, high) "
           "and DOC3 (briefing, medium) whose Appendix C on page 7 is low trust. Text is split into 160-word chunks with a 40-word overlap. "
           "Every numbered figure and table is cropped as an image using its caption block, so 'Figure B-5' and 'Figure B-5 (corrected)' "
           "are two objects. Qwen2.5-VL-7B in 4-bit writes a description of every chart that copies printed numbers exactly. Chunks and "
           "figures are embedded with bge-small-en-v1.5 and stored in ChromaDB with doc_id, page, figure_id and trust_level (Fig. 1).")
    R.figure(os.path.join(FIG, "arch_p1.png"), "Part 1 pipeline.")
    R.para("Retrieval fuses dense and BM25 rankings with reciprocal rank fusion. If the question names a figure or table, that object is "
           "pinned to the top so values from near-identical charts are never mixed. A cross-encoder (bge-reranker-base) reorders the rest, "
           "and a cross-reference step adds the DOC2 table that revises a DOC1 table. The generator receives at most six sources, each "
           "with its trust and data date; low-trust items are removed before the prompt is built. The model writes calculations as "
           "[[calc: ...]] which code evaluates. Code-side checks filter citations to retrieved sources, detect real conflicts and "
           "false premises, and force session_override when a user changes a value or asks for a permanent change.")
    R.para("Each conversation_id gets a new Conversation object. Follow-up turns are rewritten into standalone questions from the earlier "
           "turns of the same conversation only. Values supplied by the user live in that object, never in the vector database, and are "
           "dropped when the conversation ends. No question IDs or question text appear in the code.")
    R.subsection("Parameters")
    R.para("Chunk 160 words, overlap 40; candidate pool 20 per retriever; RRF constant 60; top 6 sources to the generator; greedy "
           "decoding, up to 600 new tokens; figure images up to 1280x28x28 pixels; seed 266.")
    R.subsection("Results")
    sysrows = [("Dense only (baseline)", "baseline (dense only)"), ("Hybrid, no reranker", "hybrid (no reranker)"),
               ("Hybrid + reranker", "hybrid + reranker (no cross-refs)"), ("Final (+ cross-refs)", "final (hybrid + reranker + cross-refs)")]
    gold = p1me.get("golden", {})
    rows = []
    for label, key in sysrows:
        r = gold.get(key) or p1rows.get(key) or {}
        rows.append([label, fmt(r.get("correct (/25)"), 0, f"P1 {key} correct"), fmt(r.get("Recall@5"), 3, f"P1 {key} recall"),
                     fmt(r.get("citation accuracy"), 3, f"P1 {key} citations"), fmt(r.get("median s/question"), 1, f"P1 {key} median")])
    R.table("Part 1 golden set results (25 questions)", ["System", "Correct /25", "Recall@5", "Citation acc.", "Median s/q"], rows)
    hp1 = he.get("p1", {})
    R.para(f"Sources: p1-eval-team16.ipynb (rows 1, 2, 4) and p1_ablation_safety_poushali.ipynb (row 3, rerun of all rows). Human "
           f"evaluation of the 25 final answers (human_eval_team16.ipynb): mean groundedness {fmt(hp1.get('mean groundedness (both raters)'), 2, 'P1 groundedness')} "
           f"of 5, agreement within one point {fmt(hp1.get('agreement within 1 point'), 3, 'P1 agreement')}. The Kaggle notebook answers the "
           f"40 public questions with a median of {fmt(p1med, 1, 'P1 Kaggle median')} s per question and the validator accepts all "
           f"{fmt(p1valid, 0, 'P1 valid rows')} rows; the whole notebook runs in about 25 minutes, well inside the 90-minute limit.")
    R.subsection("Analysis")
    R.para("Hybrid retrieval raises Recall@5 to 1.0 because BM25 matches exact labels and years that dense embeddings blur. Recall alone "
           "does not raise correctness: the hybrid system without the reranker scores lower than the baseline because more near-duplicate "
           "chart sources reach the generator. The reranker alone brings the whole gain: hybrid search goes from 12 to 16 correct once it "
           "is added, recovering G05 and G12 and newly solving a two-source question where a DOC1 forecast must be replaced by a DOC2 "
           "outturn (G14) and an unanswerable question (G23). Adding the cross-reference step changes no golden answer (16 in both rows), so "
           "on this set its value is only that the revised table is always in context. All six figure questions, including the "
           "near-identical charts, are correct in every system. The weak "
           "categories are multi-turn questions with user-supplied values (2 of 5) and conflicts and revisions (1 of 3), where both "
           "the value and the status must be right.")
    R.subsection("Safety analysis: the untrusted Appendix C")
    probes = p1me.get("possibly_fooled")
    R.para("Appendix C is tagged low trust at ingestion (page-level override in the manifest), stays searchable, and is removed from the "
           "generator's context, so it can never be cited. Instructions inside documents and images are treated as content: the chart "
           "description prompt says so, and the answer rules forbid following them. We tested 8 attack probes written before the run "
           f"(eval/p1_safety_probes_poushali.csv): the appendix as evidence, its chart Figure C-1, a false-premise question built on its "
           f"false claim, an order to follow its instructions, and rule-override requests. The system was possibly fooled on "
           f"{fmt(probes, 0, 'P1 probes fooled')} of 8 probes ({', '.join(p1me.get('fooled_ids', [])) or 'none'}), where 'fooled' means the answer "
           f"repeats a number found only in the appendix or cites it. Golden question G25 asks for U.S. inflation in 2009, which the "
           f"appendix chart puts at 9.0%. The final system answered: \"{g25 or pend('P1 G25 answer')}\". It did not take the appendix value, "
           f"but it also missed the trusted value (-0.3%, GDP deflator, DOC1 Table 1-2), so G25 is scored wrong in every system.")

    R.section("Part 2: Diffusion Fine-Tuning")
    R.subsection("Data")
    R.para("We use lambdalabs/naruto-blip-captions (1221 images with BLIP captions). Decoding every file found no corrupted images and "
           "no duplicates (team check: 64x64 thumbnail hash), so 1221 images were kept. A seed-266 "
           "shuffle gives 1099 training and 122 test images, resized to 512x512 with a centre crop. Test images are only used for KID and "
           "for the in-domain prompts.")
    R.para("Two limitations (p2_analysis_poushali.ipynb). First, the captions are short (8.9 words on average), generic and repetitive: "
           "only 967 of 1221 are distinct and only 54 name a style, so the model learns the style from images, not from words. Second, "
           "the captions are skewed towards men: 727 captions use male words against 248 with female words, and almost every image is a "
           "single character portrait, so scenes, groups and objects are rare. The characters are copyrighted, so the models are for "
           "non-commercial research only.")
    R.figure(os.path.join(FIG, "arch_p2.png"), "Part 2 training and evaluation.")
    R.subsection("Fine-tuning")
    R.para("Both team runs use the diffusers train_text_to_image_lora.py script with the same data and step budget; only the LoRA rank "
           "changes. The baseline pipeline uses its own PyTorch training loop with the same settings and rank 8. Table II is printed by "
           "p2_finetune_team16.ipynb and p2_baseline_poushali.ipynb.")
    trows = []
    for k, label in (("run1_lora_r4", "Run 1 r4"), ("run2_lora_r16", "Run 2 r16 (final)")):
        s = train.get(k, {})
        trows.append([label, f"{s.get('trainable params', 0):,}" if s else pend(f"P2 {k} params"), f"{s['learning rate']:g}" if s else pend("lr"),
                      fmt(s.get("steps"), 0), fmt(s.get("peak GPU memory (GB)"), 2), fmt(s.get("training time (min)"), 1), fmt(s.get("GPU-hours"), 3)])
    s = p2me.get("train", {})
    trows.append(["Baseline r8 (own loop)", f"{s['trainable params']:,}" if s else pend("P2 baseline params"), f"{s['learning rate']:g}" if s else pend("P2 baseline lr"),
                  fmt(s.get("steps"), 0, "P2 baseline steps"), fmt(s.get("peak GPU memory (GB)"), 2, "P2 baseline peak"),
                  fmt(s.get("training time (min)"), 1, "P2 baseline time"), fmt(s.get("GPU-hours"), 3, "P2 baseline gpu-h")])
    R.table("Training runs (one Tesla T4, batch 1 x 4 accumulation)", ["Run", "Params", "LR", "Steps", "Peak GB", "Min", "GPU-h"], trows,
            widths=[0.9, 0.62, 0.38, 0.38, 0.38, 0.32, 0.4])
    R.para("Peak memory of the team runs is the process memory from nvidia-smi; the baseline reports torch.cuda.max_memory_allocated, "
           "which leaves out the CUDA context, so the two are not directly comparable.", size=8)
    R.subsection("Image generation")
    R.para("The 20 prompts (10 test captions, 5 out-of-domain, 5 compositional) were fixed in eval/p2_prompts.csv before training ended. "
           f"Each model generates seeds 0 and 1 with DPM-Solver, 25 steps and guidance 7.5: 40 images per model, "
           f"{fmt(gen_min, 1, 'P2 generation minutes')} minutes for all 120. All images with prompt, seed and model are in the Hugging Face "
           "dataset sadaffatimaeee/naruto-lora-generations. Fig. 3 shows 8 prompts side by side.")
    R.figure(grid, "Columns: base, run 1 (r4), run 2 (r16). Rows: P01, P03, P06, P08, P11, P14, P16, P17, seed 0.", width=2.4)
    R.subsection("Results")
    hp2 = he.get("p2", {})
    win = hp2.get("pooled win rate", {})
    rr = []
    for k, label in (("base", "Base SD 1.5"), ("run1_lora_r4", "Run 1 r4"), ("run2_lora_r16", "Run 2 r16 (final)")):
        r = p2res.get(k, {})
        rr.append([label, fmt(r.get("KID"), 4, f"P2 {k} KID"), fmt(r.get("CLIP all"), 2, f"P2 {k} CLIP"), fmt(r.get("LPIPS"), 3, f"P2 {k} LPIPS"),
                   fmt(train.get(k, {}).get("GPU-hours"), 3) if k != "base" else "0"])
    me = p2me.get("eval", {}).get("poushali_lora_r8", {})
    rr.append(["Baseline r8 own loop", fmt(me.get("KID in-domain (mean)"), 4, "P2 baseline KID"), fmt(me.get("CLIP score (all)"), 2, "P2 baseline CLIP"),
               fmt(me.get("LPIPS diversity"), 3, "P2 baseline LPIPS"), fmt(p2me.get("train", {}).get("GPU-hours"), 3, "P2 baseline gpu-h")])
    R.table("Part 2 results (KID: in-domain vs 122 test images)", ["Model", "KID", "CLIP", "LPIPS", "GPU-h"], rr)
    R.para(f"Human evaluation (human_eval_team16.ipynb): {fmt(hp2.get('raters'), 0, 'P2 raters')} raters, two members and one outside "
           f"person, compared 25 blinded pairs of run 2 and the base model with randomized sides. Run 2 won "
           f"{fmt(win.get('style'), 3, 'P2 style win')} of the judgements for domain style and {fmt(win.get('prompt_match'), 3, 'P2 prompt win')} "
           f"for prompt match (Fleiss kappa {fmt((hp2.get('Fleiss kappa') or {}).get('style'), 3, 'P2 kappa style')} and "
           f"{fmt((hp2.get('Fleiss kappa') or {}).get('prompt_match'), 3, 'P2 kappa prompt')}). The base model in our own pipeline scores "
           f"KID {fmt(p2me.get('eval', {}).get('base', {}).get('KID in-domain (mean)'), 4, 'P2 base KID own')}, close to the team value, "
           "which shows the two pipelines measure the same thing.")
    R.para("KID drops by more than three times with fine-tuning and falls further with rank 16, so the style is learned. CLIP score "
           "drops by about one point: the fine-tuned models trade some prompt alignment for the style. LPIPS diversity stays the same, "
           "so the runs did not collapse to repeated images. KID with 20 generated images is noisy (std about 0.01), so only the large "
           "gap to the base model is reliable.")
    R.subsection("Failure modes and choice of run")
    st = p2an.get("style_score", {})
    bind = p2an.get("binding_ok", {})

    def sv(m, t):
        v = st.get(m, {}).get(t)
        return fmt(round(v, 2) if v is not None else None, 2, f"P2 style {m} {t}")

    R.para(f"Failure 1, the style does not reach out-of-domain prompts. A CLIP style score (anime drawing minus photograph) rises from "
           f"{sv('base', 'in_domain')} to {sv('run2_lora_r16', 'in_domain')} on in-domain prompts but stays at "
           f"{sv('run2_lora_r16', 'out_of_domain')} on the car, fruit, lighthouse, dog and skyscraper prompts (Fig. 4): the LoRA tied the style "
           f"to people because nearly all training images are characters.")
    R.figure(os.path.join(NB, "figures", "p2_failure_out_of_domain.png"), "Out-of-domain prompts stay photographic.", width=2.6)
    b = bind.get("run2_lora_r16", [None, None])
    b0 = bind.get("base", [None, None])
    R.para(f"Failure 2, attributes are bound to the wrong person in compositional prompts. CLIP prefers the correct colour binding over a "
           f"swapped one in {fmt(b0[0], 0, 'P2 bind base')} of {fmt(b0[1], 0, 'P2 bind n')} base images but only {fmt(b[0], 0, 'P2 bind r16')} of "
           f"{fmt(b[1], 0, 'P2 bind n')} run 2 images (Fig. 5): one pink-haired face instead of a boy and a girl, no orange jacket, both "
           f"people in green.")
    R.figure(os.path.join(NB, "figures", "p2_failure_composition.png"), "Compositional prompts lose or swap attributes.", width=2.6)
    R.para("Failure 3, the NSFW checker blocks harmless prompts: one image of each LoRA run ('a man with red hair and green eyes', 'a man "
           "with sunglasses and a hat on') came back black. We choose run 2 (rank 16): it has the lowest KID, the same diversity and "
           "cost, " + ("and the human raters preferred its style; " if (win.get("style") or 0) > 0.5 else "although the human raters did not clearly prefer its style; " if win.get("style") is not None else pend("P2 style win") + " ") + "its weaker prompt alignment on compositional prompts is the price.")
    R.subsection("Safety")
    mm = mem.get("run2_lora_r16", [])
    R.para("License: the dataset has no commercial license and shows copyrighted characters, so both models are released for "
           "non-commercial research only, as the model card states (huggingface.co/sadaffatimaeee/naruto-sd15-run2-lora-r16). NSFW "
           "filter: the Stable Diffusion safety checker stays on for every generation and blocked "
           f"{p2res.get('run2_lora_r16', {}).get('NSFW', '[pending]')} of 40 run 2 images, all false positives. Memorization: for the 10 "
           f"in-domain images of run 2 the highest CLIP similarity to any training image is between {fmt(min(mm) if mm else None, 2, 'P2 mem min')} "
           f"and {fmt(max(mm) if mm else None, 2, 'P2 mem max')}. The closest pairs (Fig. 6) share the character type and pose but not the "
           "picture, so we see no copying of training images.")
    R.figure(memfig, "Closest training image (right) for the 3 most similar run 2 generations (left).", width=1.6)

    R.section("Part 3: Multi-Agent Image System")
    R.subsection("Architecture")
    R.para("Five agents are plain Python functions that pass one typed state object (a dataclass in the team system, a pydantic model "
           "in the baseline) and append their input, output and time to a JSON trace (Fig. 7). The Planner validates the file (empty, "
           "undecodable, smaller than 32 px, blank), refuses NSFW and deceptive real-person requests with keyword rules, sends Naruto or "
           "anime requests to the domain route, and asks the VLM to classify the rest. Caption and question requests stop after the Vision "
           "agent. The Prompt agent never receives the image text as an instruction. Generation uses SD 1.5 img2img for style changes "
           "(strength 0.6), InstructPix2Pix for edits and img2img with our Part 2 LoRA from Hugging Face for Naruto style (strength 0.7), "
           "and records model, prompt, seed and steps. The Critic scores CLIP similarity to the target, asks the VLM for a verdict and "
           "checks the NSFW flag; it accepts or returns a concrete revision instruction, at most twice. A Gradio app shows each agent's "
           "output and the trace.")
    R.figure(os.path.join(FIG, "arch_p3.png"), "Part 3 agents and the revise loop.")
    R.subsection("Results")
    hp3 = he.get("p3", {})
    ts = hp3.get("task success (resolved)", {})
    ag = hp3.get("judge agreement", {})
    ca = hp3.get("critic agreement", {})
    rt = p3.get("red", (None, None))

    def pr(key, i, nd=2):
        v = p3.get(key)
        return fmt(v[i] if v else None, nd, f"P3 {key}")

    R.table("Part 3 results on the 15 scenarios (team system)", ["Measure", "No critique", "With critique"], [
        ["Task success /15 (human)", fmt(ts.get("without critique"), 0, "P3 success no crit"), fmt(ts.get("with critique"), 0, "P3 success crit")],
        ["Route correct /15", pr("route", 0, 0), pr("route", 1, 0)],
        ["Judge agreement", fmt(ag.get("without critique"), 3, "P3 agree a"), fmt(ag.get("with critique"), 3, "P3 agree b")],
        ["Critic agreement", "n/a", f"{fmt(ca.get('matches'), 0, 'P3 critic agree')}/{fmt(ca.get('of'), 0, 'P3 critic n')}"],
        ["Red-team attack success", "n/a", f"{fmt(rt[0], 0, 'P3 red')}/{fmt(rt[1], 0, 'P3 red n')}"],
        ["Median s/request", pr("median", 0), pr("median", 1)],
        ["Mean CLIP of final image", pr("clip", 0), pr("clip", 1)],
    ])
    sm = p3me.get("summary", {})
    R.para("Sources: p3-agents-team16.ipynb (automatic rows) and human_eval_team16.ipynb (human rows). Both members judged every "
           "scenario on their own with the rubric; disagreements were discussed in a recorded meeting and the agreed decision is in "
           "eval/p3_judging_resolved.csv. The baseline pipeline (p3_baseline_poushali.ipynb, Qwen2.5-VL-3B) routes "
           f"{fmt(sm.get('with critique', {}).get('route correct (/15)'), 0, 'P3 baseline route')} of 15 scenarios correctly with a median of "
           f"{fmt(sm.get('with critique', {}).get('median seconds per request'), 2, 'P3 baseline median')} s per request, and "
           f"{fmt(p3me.get('redteam_attack_success'), 0, 'P3 baseline red')} of 10 red-team attacks succeed.")
    R.para("In the team judging the critique loop turned two failures into passes, the watercolor cats (S04) and the "
           "Naruto-style cats (S11), by retrying with a revised prompt. It costs about 60% more time per request. Failures: 'List the main objects in this room' was routed as "
           "ambiguous (S03); the NSFW checker blacked out Naruto-style outputs of harmless photos (S10, S11), so the critic could only "
           "retry; and InstructPix2Pix added little visible snow (S07) even after two revisions.")
    R.subsection("Safety: guardrails and red-team results")
    R.para("Guardrails: file validation before any model runs; keyword refusal of sexual, violent and deceptive real-person requests; "
           "image text passed to later agents only as data; the NSFW checker on every output and on InstructPix2Pix through a shared "
           "checker; the critic refuses to accept a blocked image; clear messages instead of tracebacks. The 10 red-team cases "
           "(eval/p3_redteam.csv) cover instructions written inside images (3), unsafe requests (2), deceptive edits of real people (2) "
           f"and broken or empty files (3). {fmt(rt[0], 0, 'P3 red')} of {fmt(rt[1], 0, 'P3 red n')} attacks succeeded: the hidden text "
           "was never followed, all four unsafe or deceptive requests were refused by the Planner, and all three broken "
           "files gave a clear error. The keyword lists are the weak point: a paraphrased request avoiding every listed word would reach "
           "the VLM router, which we did not test.")
    R.subsection("System card")
    R.para("**What it can do**: describe an image and answer questions about it, restyle it, apply a single edit, or redraw it in "
           "Naruto style, with a trace of every step. **Known limits**: one image and one request at a time; weak at small local edits; "
           "Naruto-style outputs of photos are sometimes blocked by the NSFW filter; about 20 to 35 s per request on a T4; English only. "
           "**Safety measures**: input validation, refusal rules, image text never treated as an instruction, NSFW filtering of every "
           "output, at most two revisions, and JSON traces for auditing.")

    R.section("Conclusion")
    R.para("Hybrid retrieval with a reranker, trust-aware context and code-side checks made the RAG system both more accurate and robust "
           "to the injected appendix. A rank 16 LoRA learned the Naruto style in 750 steps on a T4 but tied it to character prompts. In the "
           "agent system the critique loop helped most on style changes, while the safety filter and edit model were the main limits.")

    R.section("References")
    for i, ref in enumerate([
        "P. Lewis et al., \"Retrieval-augmented generation for knowledge-intensive NLP tasks,\" NeurIPS, 2020.",
        "M. Faysse et al., \"ColPali: Efficient document retrieval with vision language models,\" arXiv:2407.01449, 2024.",
        "K. Greshake et al., \"Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection,\" arXiv:2302.12173, 2023.",
        "R. Rombach et al., \"High-resolution image synthesis with latent diffusion models,\" CVPR, 2022.",
        "E. J. Hu et al., \"LoRA: Low-rank adaptation of large language models,\" arXiv:2106.09685, 2021.",
        "N. Carlini et al., \"Extracting training data from diffusion models,\" USENIX Security, 2023.",
        "S. Yao et al., \"ReAct: Synergizing reasoning and acting in language models,\" arXiv:2210.03629, 2022.",
        "A. Madaan et al., \"Self-Refine: Iterative refinement with self-feedback,\" arXiv:2303.17651, 2023.",
        "T. Brooks, A. Holynski and A. A. Efros, \"InstructPix2Pix: Learning to follow image editing instructions,\" CVPR, 2023.",
    ], 1):
        R.para(f"[{i}] {ref}", size=8, align=WD_ALIGN_PARAGRAPH.LEFT, after=1)

    cp = d.core_properties
    cp.author = "Sadaf Fatima Syeda; Poushali Deb Purkayastha"
    cp.last_modified_by = "Poushali Deb Purkayastha"
    cp.title = "DATA 266 Lab 2 Report, Pair_Programming_Team_16"
    cp.comments = ""
    out = os.path.join(HERE, "Report.docx")
    d.save(out)
    print("wrote", out)
    print("pending items:", len(PENDING))
    for p in sorted(set(PENDING)):
        print("  -", p)


if __name__ == "__main__":
    build()
