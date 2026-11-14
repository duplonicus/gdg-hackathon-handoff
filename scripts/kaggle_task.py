# Kaggle Benchmarks task (push: scripts/kaggle.sh b t push handoff-decision -f scripts/kaggle_task.py -d duplonicus/handoff-frames):
# one logged robot decision (camera frame + the exact prompt our local arms get),
# answered with a choice and a self-stated confidence. Run files are converted to arms.jsonl rows at home.
import glob
import json
import os
from dataclasses import dataclass

import pandas as pd

import kaggle_benchmarks as kbench
from kaggle_benchmarks.content_types import images


@dataclass
class Answer:
    answer: str
    confidence: float


CSV = glob.glob("/kaggle/input/**/decisions.csv", recursive=True)[0]
ROOT = os.path.dirname(CSV)


@kbench.task(name="handoff-decision", description="A sorting robot's logged decision from its camera frame, with stated confidence")
def handoff_decision(llm, log: str, index_global: int, episode: str, question: str, prompt: str, frame: str, options: str, truth: str) -> dict:
    opts = json.loads(options)
    with kbench.chats.new(f"{log}-{index_global}"):
        out = llm.prompt(prompt, image=images.from_path(os.path.join(ROOT, frame)), schema=Answer)
    answer = out.answer.strip().lower()
    return {"log": log, "index_global": int(index_global), "episode": str(episode), "question": question, "answer": answer, "confidence": float(out.confidence),
            "valid": answer in opts, "truth": truth, "right": answer == truth}


df = pd.read_csv(CSV)
# Every push makes Kaggle "validate" by running the whole task on its default model (gemini-3.8-flash); on 460
# frames that is ~$2 of the free daily quota. The run's model comes from LLM_DEFAULT, so the validation model only
# answers a few rows. Any other model (the real runs) answers all of them.
if "gemini-3.8-flash" in os.environ.get("LLM_DEFAULT", ""):
    df = df.head(8)
runs = handoff_decision.evaluate(llm=[kbench.llm], evaluation_data=df, n_jobs=4, timeout=180, max_attempts=3, retry_delay=15)
print(runs.as_dataframe().to_string())
