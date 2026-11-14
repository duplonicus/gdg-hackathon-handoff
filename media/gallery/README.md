# Gallery

Stills from two recorded demo runs of the MH3 sorting cell (seed 7, `scripts/demo.py --scales --fault wrong_bin`), three frames from the film footage (Nov 7, Phase 1 and Phase 2), plus three plots from the results notebook.

The stills are the robot's own decision-camera frames (480 x 300), the exact images d3 27B was shown for each decision. They come from `output/demo/20261023-190856/frames/` (Oct 23 run, 01 to 05) and `output/demo/20261021-153304/frames/` (Oct 21 run, 06 to 08), whose `run.log` and `decisions.jsonl` give the answers quoted below. The ladder: d3 acts at 0.80 or more, otherwise a person decides on the operator page; in these runs the tray load cells answer the "did it land in the right tray" check.

| File | What it shows |
|---|---|
| `01_inspect_unpainted_part_d3_says_defective.png` | Paint inspection of part_15: grey bare-metal faces with blue paint only on the edges. d3 answered "defective" at 1.00 and the robot acted on it. |
| `02_grasp_check_open_hand_beside_part.png` | Grasp check: the robot's open left hand above the table, the grey-and-blue part_15 below it, another blue part to the left, the green good tray at right. d3 said "grasp" at 0.87, so the robot went ahead. |
| `03_wrong_bin_fault_defective_part_in_good_tray.png` | The forced wrong-bin fault: defective part_15, meant for the orange rework tray, lying in the green good tray. The load cells read the good tray up one part and the rework tray unchanged, and answered "no (other bin)". |
| `04_after_operator_rebin_part_in_rework_tray.png` | The same view after the operator moved the part: part_15 now in the orange rework tray, the green tray empty, the next blue part still on the table at top right. The load cells then answered "yes". |
| `05_inspect_scratched_part_handed_to_person.png` | Paint inspection of part_10: blue all over with a few faint light scratch lines. d3 leaned "defective" at only 0.60, under 0.80, so the call went to the person on the operator page, who answered "defective". |
| `06_inspect_mostly_unpainted_part.png` | Paint inspection of part_12: mostly grey bare metal with blue bands at the top and bottom edges. d3 answered "defective" at 0.99. |
| `07_grasp_check_handed_to_person.png` | Grasp check on part_12, the open hand above it and the green tray at right. d3 said "grasp" at 0.74, under 0.80, so the person was asked and approved the grasp. |
| `08_sort_complete_three_parts_in_rework_tray.png` | End of the Oct 21 run: all three defective parts in the orange rework tray (one leaning against the tray wall), the green good tray empty. d3 leaned "not finished" at 0.78, under 0.80, so the person decided and answered "finished". |
| `09_capsule_cell_leaker_dropped_in_bin.png` | Phase 2, the capsule cell (seed 5017, `scripts/film_shift.py`, close-up camera): the robot lets go of a leaking capsule inside the right-hand red leaker bin. The flat pools on the parchment mark where leakers were picked. All 3 leakers of that shift were binned (media/film/FOOTAGE.md). |
| `10_unsure_finish_call_answered_on_operator_page.png` | Phase 1 film take P1-B (`output/demo/20261107-200537/run.log`): left, the sim; right, the real operator page redrawn from what the page served. d3 leaned "defective" at 0.58, under 0.80, so the finish call on part_10 went to the person; the pressed button is the scripted operator's answer, "defective". |
| `11_model_down_question_waits_on_operator_page.png` | Phase 1 film take P1-D (`output/demo/20261107-204625/run.log`): d3 was stopped mid-run, so the page reads "could not answer (a model is down or failed): d3-27b-vision ERROR" and the robot holds. Nobody answered this one; after 120 s the safe answer, "defective", was taken. |
| `plot_capsule_sure_and_wrong.png` | From the notebook, Phase 2: on test05's 467 leaker pictures, how many each model called good while 0.9 or more sure. The two VDMs: 1 each. Qwen3.8-27B, the VLM d3 is built on: 98 (first-word probability) and 171 (stated confidence). |
| `plot_threshold_tradeoff.png` | From the notebook: on test04's judgement questions, how acted-wrong falls and hand-offs to the person rise as the threshold moves, with the frozen 0.80 marked (2.6% acted wrong, 18.5% handed off). An illustration, not a refit. |
| `plot_wrong_bin_blind_spot.png` | From the notebook: d3's "it's in the tray" probability on every test04 `in_bin` decision whose truth was "no". The 29 wrong-tray decisions all sit at 0.945 or more, past any useful threshold, while parts in no tray spread out and are mostly caught. |

The three plots are written by `scripts/execute_notebook.py notebooks/decision_ladder_results.ipynb --figdir media/gallery`; the stills are plain copies of the frames named above.
