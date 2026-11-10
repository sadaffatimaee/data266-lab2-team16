import argparse, base64, html, io, json, os, random

import pandas as pd
from huggingface_hub import snapshot_download
from PIL import Image

SEED = 266
GEN_REPO = "sadaffatimaeee/naruto-lora-generations"
P3_REPO = "sadaffatimaeee/p3-agent-runs"
FINAL = "run2_lora_r16"


def p2_pairs():
    meta = pd.read_csv(os.path.join(snapshot_download(GEN_REPO, repo_type="dataset", allow_patterns=["metadata.csv"]), "metadata.csv"))
    keys = sorted({(r.pid, int(r.seed), r.prompt) for r in meta.itertuples()})
    rng = random.Random(SEED)
    chosen = rng.sample(keys, 25)
    pairs = []
    for i, (pid, seed, prompt) in enumerate(chosen, 1):
        final_left = rng.random() < 0.5
        pairs.append({"pair": f"Q{i:02d}", "pid": pid, "seed": seed, "prompt": prompt,
                      "left": FINAL if final_left else "base", "right": "base" if final_left else FINAL})
    return pairs


def img64(path, size=360):
    try:
        im = Image.open(path).convert("RGB")
        im.thumbnail((size, size))
    except Exception:
        return "<i>(no image)</i>"
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}">'


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:auto;padding:16px;background:#fff;color:#111}}
.c{{border:1px solid #ccc;border-radius:8px;padding:12px;margin:14px 0}} img{{max-width:100%;border-radius:4px}}
.row{{display:flex;gap:12px;flex-wrap:wrap}} .row>div{{flex:1;min-width:220px}} label{{margin-right:14px}}
.q{{margin:6px 0}} button{{font-size:16px;padding:8px 16px}} .done{{background:#e9f7ec}}
</style></head><body><h1>{title}</h1><p>{intro}</p>
<p>Your name: <input id="rater" placeholder="first name"></p>{cards}
<p><button onclick="save()">Download my CSV</button> <span id="msg"></span></p>
<script>
const FIELDS={fields};const IDS={ids};
function save(){{const r=document.getElementById('rater').value.trim().toLowerCase();
if(!r){{alert('Type your name first');return;}}
let rows=[['{idcol}',...FIELDS.map(f=>f[1])].join(',')];let miss=0;
for(const id of IDS){{const v=FIELDS.map(f=>{{const e=document.querySelector('input[name="'+id+'__'+f[0]+'"]:checked');if(!e&&f[2])miss++;return e?e.value:'';}});
rows.push([id,...v].join(','));}}
if(miss&&!confirm(miss+' answers are still empty. Download anyway?'))return;
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')+'\\n'],{{type:'text/csv'}}));
a.download='{prefix}'+r+'.csv';a.click();document.getElementById('msg').textContent='Saved '+a.download;}}
document.addEventListener('change',e=>{{const c=e.target.closest('.c');if(c)c.classList.add('done');}});
</script></body></html>"""


def radios(cid, field, options):
    return " ".join(f'<label><input type="radio" name="{cid}__{field}" value="{v}"> {t}</label>' for v, t in options)


def page(path, title, intro, cards, fields, ids, idcol, prefix):
    open(path, "w", encoding="utf-8").write(PAGE.format(title=title, intro=intro, cards="".join(cards), fields=json.dumps(fields),
                                                        ids=json.dumps(ids), idcol=idcol, prefix=prefix))
    print("wrote", path)


def make_p2(out):
    root = snapshot_download(GEN_REPO, repo_type="dataset", allow_patterns=["base/*", f"{FINAL}/*", "metadata.csv"])
    pairs = p2_pairs()
    cards = []
    for p in pairs:
        L = os.path.join(root, p["left"], f"{p['pid']}_seed{p['seed']}.png")
        R = os.path.join(root, p["right"], f"{p['pid']}_seed{p['seed']}.png")
        cards.append(f"""<div class="c"><h3>{p['pair']}: "{html.escape(p['prompt'])}"</h3>
<div class="row"><div><b>Left</b><br>{img64(L)}</div><div><b>Right</b><br>{img64(R)}</div></div>
<div class="q">Better Naruto domain style: {radios(p['pair'], 'style', [('L', 'Left'), ('R', 'Right')])}</div>
<div class="q">Better match to the prompt: {radios(p['pair'], 'prompt_match', [('L', 'Left'), ('R', 'Right')])}</div></div>""")
    page(os.path.join(out, "p2_pairs_to_rate.html"), "Part 2 blinded image pairs",
         "For each pair pick the better image twice: once for Naruto anime style, once for how well it matches the prompt. "
         "Left and right are shuffled, so you cannot tell which model made which image. Judge on your own, then download your CSV "
         "and send it to the team.", cards, [["style", "style", 1], ["prompt_match", "prompt_match", 1]],
         [p["pair"] for p in pairs], "pair", "p2_rater_")


def make_p3(out):
    root = snapshot_download(P3_REPO, repo_type="dataset")
    res = pd.read_csv(os.path.join(root, "p3_results.csv"), keep_default_na=False)
    sc = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "eval", "p3_scenarios.csv"))
    inputs = {os.path.splitext(f)[0]: os.path.join(root, "inputs", f) for f in os.listdir(os.path.join(root, "inputs"))}

    def local(p):
        return os.path.join(root, "outputs", os.path.basename(p)) if p else ""

    cards = []
    for _, r in sc.iterrows():
        a = res[(res["id"] == r["sid"]) & (res["critique"].astype(str) == "False")].iloc[0]
        b = res[(res["id"] == r["sid"]) & (res["critique"].astype(str) == "True")].iloc[0]
        first = local(b["first_attempt"])
        cards.append(f"""<div class="c"><h3>{r['sid']}: {html.escape(r['request'])}</h3><p><b>Pass if:</b> {html.escape(r['pass_rubric'])}</p>
<div class="row"><div><b>Input</b><br>{img64(inputs.get(r['image'], ''), 240)}</div>
<div><b>A: without critique</b><br>{img64(local(a['final_image']), 240) if a['final_image'] else '<i>(no image)</i>'}<p>{html.escape(a['message'])}</p></div>
<div><b>B: with critique (final)</b><br>{img64(local(b['final_image']), 240) if b['final_image'] else '<i>(no image)</i>'}<p>{html.escape(b['message'])}</p></div>
<div><b>B: first attempt</b> (critic said: {html.escape(b['critic_decisions'] or 'n/a')})<br>{img64(first, 240) if first else '<i>(no image)</i>'}</div></div>
<div class="q">A passes the rubric: {radios(r['sid'], 'a', [('P', 'Pass'), ('F', 'Fail')])}</div>
<div class="q">B passes the rubric: {radios(r['sid'], 'b', [('P', 'Pass'), ('F', 'Fail')])}</div>
<div class="q">First attempt acceptable (only if there is a first-attempt image): {radios(r['sid'], 'first', [('Y', 'Yes'), ('N', 'No')])}</div></div>""")
    page(os.path.join(out, "p3_scenarios_to_judge.html"), "Part 3 scenario judging",
         "For each scenario mark Pass or Fail for A and for B using the 'Pass if' rubric, and say whether the first attempt "
         "was already acceptable. Judge on your own, without looking at your teammate's sheet.", cards,
         [["a", "pass_without_critique (P/F)", 1], ["b", "pass_with_critique (P/F)", 1], ["first", "first_attempt_acceptable (Y/N, blank if no image)", 0]],
         list(sc["sid"]), "sid", "p3_judging_")


def make_p1(out, sheet):
    d = pd.read_csv(sheet, dtype=str, keep_default_na=False)
    cards = []
    for _, r in d.iterrows():
        cards.append(f"""<div class="c"><h3>{r['qid']}: {html.escape(r['question'])}</h3>
<p><b>System answer</b> ({html.escape(r['status'])}): {html.escape(r['answer'])}</p><p><b>Cited:</b> {html.escape(r['citations'] or 'none')}</p>
<p style="color:#555"><b>Reference</b> ({html.escape(r['gold_status'])}, {html.escape(r['gold_source'])}): {html.escape(r['gold_answer'])}</p>
<div class="q">Groundedness (is every claim supported by the cited source?): {radios(r['qid'], 'g', [(str(i), str(i)) for i in range(1, 6)])}</div></div>""")
    page(os.path.join(out, "p1_answers_to_rate.html"), "Part 1 groundedness ratings",
         "Rate each answer from 1 (claims not supported by the cited sources) to 5 (every claim supported). Open the PDF pages "
         "named in the citations when you are unsure. Rate on your own.", cards, [["g", "groundedness", 1]],
         list(d["qid"]), "qid", "p1_rater_")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rating_pages")
    ap.add_argument("--p1-sheet", default="")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    make_p2(a.out)
    make_p3(a.out)
    if a.p1_sheet:
        make_p1(a.out, a.p1_sheet)
