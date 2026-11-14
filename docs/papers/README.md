# Papers

Every paper this project takes an idea from, with the PDF beside this file. One entry per paper: what it is, what we took or tested, and what came of it. Add new papers here when they are downloaded, and cite the entry wherever its idea is used (code comment, PAPER.md, VALIDATION.md).

## Make π0.5 Learn to Call Astra for Reliable Robot Manipulation

- Jianli Sun, Chengxiang Li, Yonglin Tian. arXiv 2610.14117, 9 Oct 2026. CC BY 4.0. https://arxiv.org/abs/2610.14117
- PDF: `2610.14117-call-head-pi05-astra.pdf`
- What it is: a 480k-parameter "CALL" head reads the frozen π0.5 policy's own features and predicts when to hand control to GPT-6 Astra. Labels come from rollouts (help = 1 minus the continuation success rate). It fires on one very high score or 4 of the last 5 over the threshold, then a 6-step cooldown. LIBERO, 200 episodes: 97.5% vs 95.5% success, Astra runs 5.5% of steps.
- Relevance: it escalates to a model only, never to a person; our ladder adds the person rung. It has no comparison against a plain confidence or entropy threshold, which is the comparison our benchmark makes. The 2-point gain is 4 episodes, with no significance test.
- Tested here: nothing yet. The 4-of-5 trigger and cooldown do not transfer: our decisions are discrete (1 to 6 per question per episode, each a different part and frame), not a control stream. A learned head on the Decider's hidden states is possible but needs a GPU pass to extract them; not done.

## JevAgentHarness: Benchmarking, Deploying and Training Decision Models That Share Control with LLMs

- Kunlun Zhu, Tao Feng, Xu Han, Heng Wang, Ling Yang, Xiangru Tang, Jiaxuan You, Heng Ji. arXiv 2610.21412, 15 Oct 2026. CC BY-NC-SA 4.0. https://arxiv.org/abs/2610.21412
- PDF: `2610.21412-per-choice-thresholds.pdf`
- What it is: a decision model answers each agent choice when its confidence beats a threshold fitted per decision point: accept when confidence >= (the LLM's accuracy there) - (cost of the LLM call / value of a wrong choice). Fitted on separate calibration runs. 81 to 89% cheaper than the LLM alone, within 3 points of its success; a fixed 0.5 threshold lost 21.5 points.
- Relevance: we already fit thresholds per rung and per question on held-out episodes. The new part is that a lower rung's cut-off should depend on the next rung, not on the person.
- Tested here (Oct 16): `choose_thresholds_joint` searches all rungs' thresholds together on the same objective. It did NOT improve our ladders: on the headline ladder it was worse on every measure over five splits. Receipts in VALIDATION.md ("Joint threshold fit"); kept in the code and the report as a recorded negative result.

## Vision-Language Models as Success Detectors

- Yuqing Du, Ksenia Konyushkova, Misha Denil, Akhil Raju, Jessica Landon, Felix Hill, Nando de Freitas, Serkan Cabi (DeepMind). arXiv 2303.07280, 13 Mar 2023. arXiv non-exclusive licence: PDF kept locally (`2303.07280-vlm-success-detectors.pdf`, gitignored), not redistributed. https://arxiv.org/abs/2303.07280
- What it is: success detection as a yes/no visual question ("Did the robot successfully {task}?"), Flamingo 3B fine-tuned on human labels, against a bespoke model per domain (sim household, real Panda gear insert/remove, Ego4D). Balanced accuracy. In-domain the bespoke model wins (97 to 99% vs 82 to 96% on the robot tasks); under a new camera view or distractors the fine-tuned VLM holds up far better (drops of 4 to 8 points vs up to 48). Zero-shot Flamingo scores 50%. Used as a filter for behaviour cloning, a few points less accuracy gave about the same policy (30% vs 33%).
- Relevance: no calibration, confidence, threshold or abstention anywhere; the authors say false positives and false negatives must be characterised before using a detector as a reward. Reusable: balanced accuracy as a metric, and filtered behaviour cloning (keep only episodes the judge calls a success) as a first learning use, safer than RL.
- Tested here: nothing yet. Their data is internal, so none of it can be run here.

## FailBench: How Reliable are VLMs at Judging Robot Task Success?

- Zaruhi Navasardyan, Tatul Danielyan, Hrant Davtyan (Metric AI Lab). arXiv 2609.03611, 3 Sep 2026. arXiv non-exclusive licence: PDF kept locally (`2609.03611-failbench.pdf`, gitignored). https://arxiv.org/abs/2609.03611 · https://metric-ai-lab.github.io/failbench/
- What it is: 2,197 real and simulated tabletop manipulation attempts from 14 public sources, binary success label, 13 judges. Best balanced accuracy 0.77 (Gemini 3 Flash), Gemma-4-31B 0.75, Qwen3-VL-2B 0.53; contact-rich assembly at most 0.60. All five purpose-built failure detectors scored worse than their own base model outside their training data. General VLMs lean toward calling a failure a success. Simulation ranks judges like real data (Spearman 0.95) but hides the gaps.
- Relevance: no calibration measured for any judge; the paper calls calibration under shift an open problem, which is exactly the gap our calibrated-ladder result speaks to. Warnings for us: fine-tuning a judge on its own data can hurt it elsewhere, so any fine-tune on the ladder's labels needs a shifted test; small models sit near chance on cross-source robot video, so the Decider's value is calibration and hand-off, not raw accuracy; report the acted-wrong rate split into false success vs false failure, and a constant-answer baseline.
- Tested here: not yet. The dataset is not released (checked 17 Oct 2026: project page says pending; the Apache-2.0 GitHub repo holds only a README). When it is, run the Decider arms on it and report balanced accuracy, ECE and confidently wrong next to their Table 1.
