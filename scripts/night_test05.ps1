# The capsule confirmation night (docs/PREREG_CAPSULES.md): test05, a log no choice has touched, then the four
# FREEZE arms replayed on it, then the frozen ladder scored once (no refit), then the catch rates.
# Unattended: each step logs and the next still runs if one fails. Resumable: run it again and finished work is skipped.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\night_test05.ps1
Set-Location (Split-Path -Parent $PSScriptRoot)
$log = "output\capsules\night_test05.log"
New-Item -ItemType Directory -Force -Path "output\capsules" | Out-Null
function Note($text) { "$(Get-Date -Format s)  $text" | Out-File -FilePath $log -Append -Encoding utf8; $text }

Note "start: confirmation log test05 (150 shifts, seeds 3000-3149, oracle, lean headless)"
& .\run.ps1 scripts/run_capsules.py --episodes 150 --seed0 3000 --out output/capsules/test05 *>&1 | Out-File -Append "output\capsules\test05.log" -Encoding utf8
Note "test05 done (exit $LASTEXITCODE)"

$arms = "d3-27b-vision,intern-4b-vision,qwen-27b-vision,qwen-27b-self-vision"
Note "replays: $arms"
& .\run.ps1 scripts/replay_capsules.py --logs "output/capsules/test05" --arms $arms *>&1 | Out-File -Append "output\capsules\test05_replay.log" -Encoding utf8
Note "replays done (exit $LASTEXITCODE)"

Note "confirm: the frozen sel03 ladder on test05, scored once"
& .\run.ps1 scripts/freeze_ladder.py confirm --log output/capsules/test05 --frozen output/capsules/sel03/frozen-ladder.json *>&1 | Out-File -Append "output\capsules\test05_confirm.log" -Encoding utf8
Note "confirm done (exit $LASTEXITCODE)"

Note "catch rates"
& .\run.ps1 python scripts/capsule_catch_rates.py output/capsules/test05 *>&1 | Out-File -Append "output\capsules\test05_catch_rates.log" -Encoding utf8
Note "all done (exit $LASTEXITCODE)"
