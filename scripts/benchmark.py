r"""Replay logged decisions through each arm, then score the arms and the composed ladders.

Step 1, once per arm (each needs its model up; run them one at a time):

    bash run.sh scripts/benchmark.py replay --log output/sort/<run> --arm gemma-e4b
    bash run.sh scripts/benchmark.py replay --log output/sort/<run> --arm gemma-e2b-self
    bash run.sh scripts/benchmark.py replay --log output/sort/<run> --arm decider-2b-vision   # D:\Models\decision-servers\decider\serve.cmd (native Windows)
    bash run.sh scripts/benchmark.py replay --log output/sort/<run> --arm decider-2b-text
    bash run.sh scripts/benchmark.py replay --log output/sort/<run> --arm decider-e2b-text --url http://127.0.0.1:8766
    bash run.sh scripts/benchmark.py describe --log output/sort/<run>      # Gemma scene descriptions for the text arms

Step 2, no models needed:

    bash run.sh scripts/benchmark.py score --log output/sort/<run>
    bash run.sh scripts/benchmark.py robust --log output/sort/<run> --splits 10 --human-costs 0.2,0.35,0.5,0.65,0.8

Arms:
  gemma-e4b, gemma-e2b        Gemma alone, one-word answer, probabilities from first-token logprobs
  gemma-e2b-self              small Gemma with self-reported confidence (escalation to e4b is composed at scoring)
  gemma-e4b-self              the bigger Gemma's self-reported confidence, the top of that escalation
  decider-2b-vision           the Decider looking at the frame directly, no Gemma description
  decider-2b-text, decider-e2b-text, decider-e4b-text
                              a Decider reading the state text plus Gemma's description of the frame
  gemma-12b                   the 12B Gemma, same one-word method
  clef-flash-vision, clef-flash-text, nimble-v3-text, tev1-4b-text, tev1-0.8b-text, laya-text, kev-4b-text
                              decision models served by Ollama's /v1/systemone (OLLAMA_DECISION_MODELS below; pull the tag first)
  d3-nano-vision, intern-4b-vision
                              HF decision models behind their own servers (d3_server.py on 8768, scripts/intern_server.py on 8767)
  gemini-er2-vision           Gemini Robotics ER 2 (Google API, free tier), frame + state, self-stated confidence
  argon-vision                Gemini 4 Argon through OpenRouter (PAID), frame + state, self-stated confidence
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from handoff.benchmark import RULE_DECIDED, add_logged_arms, choose_thresholds, choose_thresholds_joint, choose_thresholds_joint_per_question, choose_thresholds_per_question, collapse_finish, compose_ladder, group_by_arm, relabel_bin_questions, rows_for_arm, rung_value, score, split  # noqa: E402
from handoff.scene import Layout  # noqa: E402
from handoff.decider import QUESTIONS  # noqa: E402
from handoff.logbook import read_decisions  # noqa: E402
from handoff.cloud import GeminiRung, OpenRouterRung  # noqa: E402
from handoff.rungs import DECIDER_URL, OLLAMA_URL, DeciderRung, GemmaRung, GemmaSelfRung, LlamaChatRung, _post, unload_ollama_models  # noqa: E402


def load(log: Path) -> list[dict]:
    decisions = read_decisions(log / "decisions.jsonl")
    for i, d in enumerate(decisions):
        d["index_global"] = i
    layout = Layout()
    changed = relabel_bin_questions(decisions, layout, layout.part_size)
    if changed:
        print(f"relabelled {changed} bin-question truths from logged positions with the current check")
    return decisions


def frame_b64(log: Path, d: dict) -> str | None:
    if not d.get("frame"):
        return None
    return base64.b64encode((log / d["frame"]).read_bytes()).decode()


DEFAULT_DESCRIBER = "gemma4:e4b"


def descriptions_path(log: Path, describer: str = DEFAULT_DESCRIBER) -> Path:
    """One file per describer, so a second describer never skips frames the first one wrote and a text arm reads the
    descriptions it was meant to (commit review round 5, Nov 4). The default keeps the old name for the old logs."""
    return log / ("descriptions.jsonl" if describer == DEFAULT_DESCRIBER else f"descriptions-{describer.replace(':', '_')}.jsonl")


def describe(log: Path, model: str = DEFAULT_DESCRIBER) -> None:
    """One short description per frame, so text-only rungs can be benchmarked on the same decisions."""
    decisions = load(log)
    out = descriptions_path(log, model)
    if model == "qwen":  # port 8774 is shared by five 27B models: never let another one write descriptions as Qwen's
        served = LlamaChatRung("qwen-describe", "http://127.0.0.1:8774", expect=("qwen3.8-27b",)).check_model()
        print(f"describer: server reports {served!r}")
    done = {int(json.loads(line)["index_global"]) for line in out.open(encoding="utf-8")} if out.exists() else set()
    schema = {"type": "object", "properties": {"description": {"type": "string"}}, "required": ["description"]}
    with out.open("a", encoding="utf-8") as handle:
        for d in decisions:
            if d["index_global"] in done:
                continue
            img = frame_b64(log, d)
            t0 = time.perf_counter()
            if model == "qwen":
                # Qwen3.8-27B on llama.cpp (dup, Nov 4: the strongest describer we can run locally, d3's own base model).
                # The question-agnostic wording fits the capsule questions; the Gemma prompt below names the sorting task.
                prompt = (f"You are the eyes of a factory robot. In two sentences, describe what this camera picture shows that matters for the question: "
                          f"{QUESTIONS[d['question']].text} Mention anything unusual. Do not answer the question.")
                content = ([{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}] if img else []) + [{"type": "text", "text": prompt}]
                body = {"messages": [{"role": "user", "content": content}], "temperature": 0, "max_tokens": 160, "chat_template_kwargs": {"enable_thinking": False},
                        "response_format": {"type": "json_schema", "json_schema": {"name": "description", "schema": schema}}}
                reply = _post("http://127.0.0.1:8774/v1/chat/completions", body, 600)
                text = json.loads(reply["choices"][0]["message"]["content"]).get("description", "")
            else:
                prompt = (
                    "You are the eyes of a sorting robot. In two sentences, describe what this camera frame shows that matters for the question: "
                    f"{QUESTIONS[d['question']].text} Mention the part's surface, its position relative to the hand, table and bins, and anything unusual. Do not answer the question."
                )
                message = {"role": "user", "content": prompt, **({"images": [img]} if img else {})}
                reply = _post(f"{OLLAMA_URL}/api/chat", {"model": model, "stream": False, "think": False, "keep_alive": "30m", "format": schema, "options": {"temperature": 0}, "messages": [message]}, 600)
                text = json.loads(reply["message"]["content"]).get("description", "")
            handle.write(json.dumps({"index_global": d["index_global"], "model": model, "description": text, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}) + "\n")
            handle.flush()
            print(f"{d['index_global']:4d} {d['question']:14s} {text[:90]}")


# Decision models served by Ollama's /v1/systemone (Ollama 0.35+). Arm name = key + "-vision" (with the frame) or "-text"
# (state text plus Gemma's description). The pull tag is the value.
OLLAMA_DECISION_MODELS = {
    "clef-flash": "clef-flash:latest",  # Cloudflare, 9B, sees images
    "nimble-v3": "nimble-v3:latest",  # Bespoke Labs, 9B, text only (pulled as nimble-v3)
    "tev1-4b": "tev1:4b",  # Together AI, text only
    "tev1-0.8b": "tev1:0.8b",
    "d1-3b": "d1:3b",  # Liquid AI, 3B, sees images (official Ollama tag, Oct 15)
    "laya": "laya",  # Convai, about 150M, text only
    "kev-4b": "hf.co/ggml-org/Kev-4B-GGUF:Q8_0",  # text only
    "d1-omni": "d1-omni",  # Liquid AI, 600M, sees images (official Ollama tag, Oct 18)
    "pplx-decider-artemis": "pplx-decider-artemis-ctx4k",  # Perplexity 27B, TurinTech Artemis ~4.5 bpw GGUF, sees images; local Modelfile at D:\Models\pplx-decider (num_ctx 4096)
}


# Three families (Oct 14): A = decision models reading the frame, no VLM anywhere; B = decision models reading
# the state text plus Gemma's description of the frame (a VLM step by construction); C = Gemma itself as a rung.
LADDERS = {
    "A frame only: decider-2b-vision > human": ["decider-2b-vision"],
    "A frame only: pplx-decider-artemis-vision > human": ["pplx-decider-artemis-vision"],
    "A frame only: decider-2b-vision > pplx-decider-artemis-vision > human": ["decider-2b-vision", "pplx-decider-artemis-vision"],
    "B description: decider-2b-text > decider-e2b-text > human": ["decider-2b-text", "decider-e2b-text"],
    "B description: decider-2b-text > decider-e4b-text > human": ["decider-2b-text", "decider-e4b-text"],
    "B description: decider-2b-vision > decider-e4b-text > human": ["decider-2b-vision", "decider-e4b-text"],
    "C gemma rung: gemma-e4b > human": ["gemma-e4b"],
    "C gemma rung: decider-2b-vision > gemma-e4b > human": ["decider-2b-vision", "gemma-e4b"],
    "C gemma rung: pplx-decider-artemis-vision > gemma-e4b > human": ["pplx-decider-artemis-vision", "gemma-e4b"],
    "C gemma rung: gemma-e2b-self > gemma-e4b-self > human": ["gemma-e2b-self", "gemma-e4b-self"],
    # D = a cloud model as a rung (Oct 15; scored with --subset-of on the sample the cloud arm answered).
    "D cloud: argon-vision > human": ["argon-vision"],
    "D cloud: decider-2b-vision > argon-vision > human": ["decider-2b-vision", "argon-vision"],
}


def run_info(**settings: object) -> dict:
    """What produced a report: the code version (truths are relabelled at load time by the current code) and the settings."""
    import subprocess

    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, timeout=20, check=False).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""

    return {"git_commit": git("rev-parse", "HEAD") or "unknown", "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "date": time.strftime("%Y-%m-%d %H:%M:%S %z"), "argv": sys.argv[1:], **settings}


def make_arm(name: str, url: str | None):
    if name == "gemma-e4b":
        return GemmaRung("gemma4:e4b", name)
    if name == "gemma-e2b":
        return GemmaRung("gemma4:e2b", name)
    if name == "gemma-12b":
        return GemmaRung("gemma4:12b", name)
    for key, tag in OLLAMA_DECISION_MODELS.items():
        if name in (f"{key}-vision", f"{key}-text"):
            return DeciderRung(name, url or OLLAMA_URL, with_image=name.endswith("-vision"), model=tag)
    if name == "gemma-e2b-self":
        return GemmaSelfRung("gemma4:e2b", name)
    if name == "gemma-e4b-self":
        return GemmaSelfRung("gemma4:e4b", name)
    if name == "gemini-er2-vision":
        return GeminiRung("gemini-robotics-er-2-preview", name)
    if name == "argon-vision":
        return OpenRouterRung("google/gemini-4-argon", name)
    if name in ("d3-nano-vision", "d3-mini-vision", "d3-edge-vision", "d3-flash-vision", "d3-lite-vision"):  # vllm-sr d3, each package's own d3_server.py (native Windows venv), port 8768
        return DeciderRung(name, url or "http://127.0.0.1:8768", with_image=True, data_url=True)
    if name in ("intern-4b-vision", "intern-2b-vision", "intern-0.8b-vision"):  # Intern-Decision 4B / 2B / 0.8B behind scripts/intern_server.py, port 8767
        return DeciderRung(name, url or "http://127.0.0.1:8767", with_image=True)
    if name == "winnow-12b-vision":  # EldanRing Winnow-12B Q8 behind its own winnow-inference server, in WSL (dup's call, Oct 18), port 8091
        return DeciderRung(name, url or "http://127.0.0.1:8091", with_image=True, image_field="winnow", timeout=120.0)
    if name == "rsi-jev-4b-vision":  # RSI-Jev v6.1 VL 4B, `rsi-jev serve` in its own venv, port 8770
        return DeciderRung(name, url or "http://127.0.0.1:8770", with_image=True, data_url=True, model="rsi-jev-v6.1-vl-4b")
    if name in ("imajev-4b-vision", "imajev-9b-vision"):  # Imajev, the repo's own playground server, port 8771
        return DeciderRung(name, url or "http://127.0.0.1:8771", with_image=True, data_url=True)
    if name in ("jpt-4b-vision", "jpt-9b-vision"):  # kirp JPT behind llm2jev's hf backend, port 8772; the frame goes inside state
        return DeciderRung(name, url or "http://127.0.0.1:8772", with_image=True, image_field="state", timeout=120.0)
    if name == "basal-omni-mini-vision":  # Remek basal-omni-1.0-mini behind basal-serve, port 8773; the frame goes in media
        return DeciderRung(name, url or "http://127.0.0.1:8773", with_image=True, image_field="media", timeout=120.0)
    # The five 27B-class arms share port 8774, so each names words its served model must hold (replay checks them first).
    if name in ("d3-27b-vision", "clef-27b-vision", "startlux-27b-vision"):  # GGUF + vision projector on llama.cpp b11723, port 8774 (D:\Models\gguf\<name>); StartLux through its maker's startlux-decide engine in front
        expect = {"d3-27b-vision": ("d3-27b",), "clef-27b-vision": ("clef",), "startlux-27b-vision": ("startlux",)}[name]
        return DeciderRung(name, url or "http://127.0.0.1:8774", with_image=True, data_url=True, timeout=120.0, expect=expect)
    if name in ("onejev-27b-vision", "onejev-9b-vision"):  # OneJev 1.6 GGUF through its maker's qev engine (llama.cpp b11723 behind it), port 8774
        return DeciderRung(name, url or "http://127.0.0.1:8774", with_image=True, image_field="qev", timeout=120.0, expect=("onejev", name.split("-")[1]))
    if name == "amos-medium-vision":  # Amos 1 medium behind its own serve/ package (amos.serve), port 8775
        return DeciderRung(name, url or "http://127.0.0.1:8775", with_image=True, data_url=True, timeout=120.0)
    if name == "pplx-flash-v12-vision":  # pplx-decider-flash-v1.2 behind scripts/pplx_server.py, port 8769
        return DeciderRung(name, url or "http://127.0.0.1:8769", with_image=True, data_url=True)
    if name in ("qwen-27b-vision", "qwen-27b-self-vision"):  # Qwen3.8-27B, d3's own base model, as a plain VLM on llama.cpp b11723, port 8774 (scripts/get_qwen.ps1)
        return LlamaChatRung(name, url or "http://127.0.0.1:8774", mode="self" if "-self" in name else "first_word", expect=("qwen3.8-27b",))
    if name == "decider-2b-vision":
        return DeciderRung(name, url or DECIDER_URL, with_image=True)
    if name.startswith("decider-") and name.endswith("-text"):
        return DeciderRung(name, url or DECIDER_URL, with_image=False)
    raise SystemExit(f"unknown arm {name}")


def host_label() -> str:
    """Where this replay runs, recorded on every arm row so latencies from different machines are never mixed unlabelled."""
    import platform

    system = platform.system()
    try:
        if "microsoft" in Path("/proc/version").read_text(encoding="utf-8").lower():
            system = "WSL"
    except OSError:
        pass
    return f"{platform.node()} {system}"


def replay(log: Path, arm: str, url: str | None, limit: int | None, same_as: str | None = None, describer: str = DEFAULT_DESCRIBER) -> None:
    unload_ollama_models()
    host = host_label()
    decisions = load(log)
    descriptions = {}
    desc_path = descriptions_path(log, describer)
    if desc_path.exists():
        descriptions = {int(json.loads(line)["index_global"]): json.loads(line)["description"] for line in desc_path.open(encoding="utf-8")}
    if arm.endswith("-text") and not descriptions:
        raise SystemExit("text arms need descriptions first: run `describe`")
    if same_as is not None:
        # Answer only the decisions another (sampled) arm answered, so the two compare and stitch on one subset.
        wanted = set(load_arms(log).get(same_as, {}))
        decisions = [d for d in decisions if int(d["index_global"]) in wanted]
        print(f"{len(decisions)} decisions, the ones {same_as} answered")
    rung = make_arm(arm, url)
    served = None
    if getattr(rung, "expect", ()):
        try:
            served = rung.check_model()
        except RuntimeError as error:
            raise SystemExit(f"{error}: stop that server before replaying {arm}") from None
        print(f"{arm}: server reports {served!r}")
    out = log / "arms.jsonl"
    # Errored rows are not done: a rerun retries them (scoring ignores them either way).
    rows = [json.loads(line) for line in out.open(encoding="utf-8")] if out.exists() else []
    # A text arm reading another describer's descriptions is its own arm: same model, different input (review round 6).
    label = f"{arm}@{describer}" if arm.endswith("-text") and describer != DEFAULT_DESCRIBER else arm
    done = {(r["arm"], int(r["index_global"])) for r in rows if not r.get("error")}
    n = errors = 0
    with out.open("a", encoding="utf-8") as handle:
        for d in decisions:
            if (label, d["index_global"]) in done:
                continue
            if limit is not None and n >= limit:
                break
            q = QUESTIONS[d["question"]]
            state = d["state_text"]
            if arm.endswith("-text"):
                state = f"{state}\nCamera: {descriptions.get(d['index_global'], '')}"
            img = None if arm.endswith("-text") else frame_b64(log, d)
            a = rung.answer(q, state, img)
            row = {"arm": label, "index_global": d["index_global"], "question": d["question"], "probs": a.probs, "latency_ms": round(a.latency_ms, 1), "error": a.error, "truth": d.get("truth"), "host": host, "server": getattr(rung, "url", url) or "", **({"served_model": served} if served else {}), **({"extra": a.extra} if a.extra else {})}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            n += 1
            errors += a.error is not None
            mark = "ok " if a.top == d.get("truth") else "XX "
            print(f"{mark}{d['index_global']:4d} {d['question']:14s} top {a.top:10s} {a.confidence:.2f} truth {d.get('truth')}  {a.latency_ms:7.0f} ms {a.error or ''}")
    print(f"{n - errors} new answers and {errors} errors for {label} in {out}")
    if errors:  # a server that died part way through must not read as a finished replay (review round 6)
        raise SystemExit(2)


def load_arms(log: Path) -> dict:
    arm_rows = [json.loads(line) for line in (log / "arms.jsonl").open(encoding="utf-8")] if (log / "arms.jsonl").exists() else []
    return group_by_arm([r for r in arm_rows if not r.get("error")])


def score_all(log: Path, held_out: float, seed: int, thresholds_from: Path | None = None, subset_of: str | None = None, human_cost: float = 0.5) -> None:
    decisions = load(log)
    arms = load_arms(log)
    folded = collapse_finish(decisions, arms)
    if folded:
        print(f"folded {folded} finish truths from scratched / unpainted into defective (log made before Oct 12)")
    for name in add_logged_arms(decisions, arms):
        print(f"{name}: scored from the readings logged with each decision (no replay); not on any ladder below")
    if thresholds_from is not None:
        # Thresholds come from a different log entirely; every decision here is test.
        held, test = load(thresholds_from), decisions
        held_arms = load_arms(thresholds_from)
        collapse_finish(held, held_arms)
        add_logged_arms(held, held_arms)
    else:
        held, test = split(decisions, held_out, seed)
        held_arms = arms
    report: dict = {"n_decisions": len(decisions), "n_held_out": len(held), "n_test": len(test), "thresholds_from": str(thresholds_from) if thresholds_from else "held-out episodes of this log", "arms": {}, "ladders": {}, "thresholds": {}, "rung_value": {}}
    report["run"] = run_info(seed=seed, held_out=held_out, human_cost=human_cost)
    if subset_of is not None:
        # Score every arm on only the decisions this arm answered (a cloud arm run on a sample), so the rows compare.
        covered = set(arms.get(subset_of, {}))
        test = [d for d in test if int(d["index_global"]) in covered]
        print(f"subset: the {len(test)} test decisions {subset_of} answered ({len({d['episode'] for d in test})} episodes)")
        report["subset_of"] = subset_of
        report["n_test"] = len(test)
    print(f"{len(decisions)} decisions; thresholds from {report['thresholds_from']} ({len(held)} decisions), {len(test)} scored\n")
    def ci(rep, key: str) -> str:
        lo, hi = (rep.interval or {}).get(key, (float("nan"), float("nan")))
        return f"[{lo:.2f} {hi:.2f}]"

    print("95% intervals are by-episode bootstrap. 'judgement' = the decisions no hard rule decides "
          f"(all questions but {', '.join(sorted(RULE_DECIDED))}); the arms sit at the base rate on the rest.")
    header = f"{'arm':28s} {'n':>4s} {'right':>6s} {'95%':>11s} {'conf.wrong':>10s} {'95%':>11s} {'ECE':>6s} {'conf':>5s} {'ms':>7s}  {'judgement n right conf.wrong':s}"
    print(header)
    for name in sorted(arms):
        rep = score(name, rows_for_arm(test, arms[name]))
        hosts = sorted({r.get("host") or "unrecorded" for r in arms[name].values()})
        report["arms"][name] = {**rep.as_dict(), "hosts": hosts}
        j = rep.judgement or {}
        print(f"{name:28s} {rep.n:4d} {rep.accuracy:6.3f} {ci(rep, 'accuracy'):>11s} {rep.confidently_wrong:10.3f} {ci(rep, 'confidently_wrong'):>11s} {rep.ece:6.3f} {rep.mean_confidence:5.2f} {rep.latency_ms:7.0f}  "
              f"{j.get('n', 0):4d} {j.get('accuracy', float('nan')):.3f} {j.get('confidently_wrong', float('nan')):.3f}   host {'; '.join(hosts)}")

    ladders = LADDERS
    print()
    for label, rung_names in ladders.items():
        if not all(r in arms for r in rung_names):
            continue
        if not all(r in held_arms for r in rung_names):
            continue
        test_ids = {int(d["index_global"]) for d in test}
        if any(len(test_ids & set(arms[r])) < 0.95 * len(test_ids) for r in rung_names):
            print(f"skipped {label}: a rung answered under 95% of the scored decisions (score it with --subset-of)")
            continue
        # "joint" modes (Oct 16): every rung's threshold searched together, so a lower rung passes up what the next
        # rung does better (arXiv 2610.21412's per-choice rule); the greedy modes compare each rung with the person.
        fitters = {"per rung": choose_thresholds, "per rung and question": choose_thresholds_per_question}
        if len(rung_names) > 1:
            fitters |= {"joint": choose_thresholds_joint, "joint, per question": choose_thresholds_joint_per_question}
        value_thresholds = None
        for mode, fit in fitters.items():
            thresholds = fit(held, [(r, held_arms[r]) for r in rung_names], human_cost=human_cost)
            if mode == "per rung and question":
                value_thresholds = thresholds
            rows = compose_ladder(test, [(r, arms[r], thresholds[r]) for r in rung_names])
            rep = score(f"{label} [{mode}]", rows)
            report["ladders"][f"{label} [{mode}]"] = rep.as_dict()
            report["thresholds"][f"{label} [{mode}]"] = thresholds
            print(f"{label} [{mode}]\n  thresholds {thresholds}")
            print(f"  {rep.n:4d} decided alone and right {rep.alone_right:.3f} {ci(rep, 'alone_right')}  handed to a person {rep.human_rate:.2f} {ci(rep, 'human_rate')}  "
                  f"conf.wrong {rep.confidently_wrong:.3f} {ci(rep, 'confidently_wrong')}  right incl. the person {rep.accuracy:.3f}  ECE {rep.ece:.3f} ms {rep.latency_ms:.0f}")
            if rep.judgement:
                j = rep.judgement
                print(f"  judgement only ({j['n']}): decided alone and right {j['alone_right']:.3f}  handed to a person {j['human_rate']:.2f}  conf.wrong {j['confidently_wrong']:.3f}")
            if rep.by_question:
                print("  per question: " + "; ".join(f"{q} right {v['accuracy']:.2f} human {v['human_rate']:.2f}" for q, v in rep.by_question.items()))
        for lower, upper in zip(rung_names[:-1], rung_names[1:], strict=True):
            v = rung_value(test, arms[lower], value_thresholds[lower], arms[upper])
            report["rung_value"][f"{lower} -> {upper}"] = v
            print(f"  escalated {lower} -> {upper}: n {v['n_escalated']}, upper right {v['upper_accuracy']:.3f} vs lower's own guess {v['lower_accuracy_on_same']:.3f}")
    tag = f"-thresholds-{thresholds_from.name}" if thresholds_from else ""
    out = log / (f"benchmark-subset-{subset_of}{tag}.json" if subset_of else "benchmark.json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwritten {out}")


def robust(log: Path, splits: int, held_out: float, seed0: int, human_costs: list[float], min_decisions: int) -> None:
    """Every ladder over many held-out splits and person costs, so no figure rests on one lucky split or one cost.

    For each ladder, threshold fit and person cost: the thresholds are fitted on each split's held-out episodes and
    the ladder scored on that split's test episodes; the report gives the mean and the range over the splits.
    Splits share decisions, so the range is a spread, not a confidence interval.
    """
    decisions = load(log)
    arms = load_arms(log)
    collapse_finish(decisions, arms)
    fits = {
        "per rung": choose_thresholds,
        "per question": choose_thresholds_per_question,
        f"per question, under {min_decisions} pooled": lambda d, r, **kw: choose_thresholds_per_question(d, r, min_decisions=min_decisions, **kw),
    }
    all_ids = {int(d["index_global"]) for d in decisions if d.get("truth") is not None}
    report: dict = {"run": run_info(splits=splits, held_out=held_out, seed0=seed0, human_costs=human_costs, min_decisions=min_decisions), "n_decisions": len(decisions), "ladders": {}}
    keys = ("alone_right", "human_rate", "confidently_wrong")  # plus acted_wrong (answered without the person, and wrong) per split
    summary_keys = (*keys, "acted_wrong", "judgement_acted_wrong", "judgement_confidently_wrong")
    print(f"{len(decisions)} decisions; {splits} splits (seeds {seed0} to {seed0 + splits - 1}), held-out fraction {held_out}; mean [min max] over splits\n")
    for label, rung_names in LADDERS.items():
        if not all(r in arms for r in rung_names) or any(len(all_ids & set(arms[r])) < 0.95 * len(all_ids) for r in rung_names):
            continue  # a rung answered only a sample (a cloud arm): score it with --subset-of instead
        print(label)
        for hc in human_costs:
            for fit_name, fit in fits.items():
                per_split = []
                for k in range(splits):
                    held, test = split(decisions, held_out, seed0 + k)
                    t = fit(held, [(r, arms[r]) for r in rung_names], human_cost=hc)
                    rep = score(label, compose_ladder(test, [(r, arms[r], t[r]) for r in rung_names]), bootstrap=0)
                    j = rep.judgement or {}
                    per_split.append({key: getattr(rep, key) for key in keys} | {"acted_wrong": round(1 - rep.alone_right - rep.human_rate, 4),
                                     "judgement_acted_wrong": round(1 - j.get("alone_right", 0) - j.get("human_rate", 0), 4) if j else None,
                                     "judgement_confidently_wrong": j.get("confidently_wrong")})
                summary = {key: {"mean": round(sum(p[key] for p in per_split) / len(per_split), 4), "min": min(p[key] for p in per_split), "max": max(p[key] for p in per_split)}
                           for key in summary_keys if all(p.get(key) is not None for p in per_split)}
                report["ladders"].setdefault(label, {}).setdefault(f"human_cost {hc}", {})[fit_name] = {"summary": summary, "per_split": per_split}
                print(f"  cost {hc:.2f} {fit_name:28s} " + "  ".join(f"{key} {summary[key]['mean']:.3f} [{summary[key]['min']:.3f} {summary[key]['max']:.3f}]" for key in (*keys, "acted_wrong") if key in summary))
    out = log / "benchmark-robust.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwritten {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("replay", "describe", "score", "robust"))
    parser.add_argument("--log", required=True)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--url", default=None, help="Decider server URL for a decider-* arm")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--held-out", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--same-as", default=None, help="replay: answer only the decisions this arm answered")
    parser.add_argument("--describer", default=DEFAULT_DESCRIBER, help="describe: who writes them (an Ollama tag, or qwen: Qwen3.8-27B on llama.cpp, port 8774); replay: whose descriptions a text arm reads")
    parser.add_argument("--subset-of", default=None, help="score on only the decisions this arm answered (writes benchmark-subset-<arm>.json)")
    parser.add_argument("--human-cost", type=float, default=0.5, help="score: what a hand-off to the person costs in the threshold objective (right +1, wrong -1, person 1 - cost)")
    parser.add_argument("--splits", type=int, default=10, help="robust: how many held-out splits")
    parser.add_argument("--human-costs", default="0.5", help="robust: comma-separated person costs to sweep, e.g. 0.2,0.35,0.5,0.65,0.8")
    parser.add_argument("--min-decisions", type=int, default=50, help="robust: a question with fewer held-out decisions takes the rung's single threshold")
    parser.add_argument("--thresholds-from", default=None, help="another log whose decisions choose the thresholds; this log is then all test")
    args = parser.parse_args()
    log = Path(args.log)
    if args.command == "describe":
        describe(log, args.describer)
    elif args.command == "replay":
        if not args.arm:
            raise SystemExit("--arm is required")
        replay(log, args.arm, args.url, args.limit, args.same_as, args.describer)
    elif args.command == "robust":
        robust(log, args.splits, args.held_out, args.seed, [float(x) for x in args.human_costs.split(",")], args.min_decisions)
    else:
        score_all(log, args.held_out, args.seed, Path(args.thresholds_from) if args.thresholds_from else None, args.subset_of, args.human_cost)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
