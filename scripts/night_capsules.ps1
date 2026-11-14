# The one GPU night for the capsule study (docs/FREEZE.md): the selection log, its block-size cuts, every model arm
# replayed on all of it, then the catch rates. Unattended: each step logs and the next still runs if one fails.
# Every step is resumable: run the same command again and finished work is skipped.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\night_capsules.ps1
Set-Location (Split-Path -Parent $PSScriptRoot)
$log = "output\capsules\night.log"
New-Item -ItemType Directory -Force -Path "output\capsules" | Out-Null
function Note($text) { "$(Get-Date -Format s)  $text" | Tee-Object -FilePath $log -Append }

Note "start: selection log sel03 (150 shifts, seeds 2000-2149, oracle, lean headless, two lamps + drop inside the bin)"
& .\run.ps1 scripts/run_capsules.py --episodes 150 --seed0 2000 --out output/capsules/sel03 *>&1 | Out-File -Append "output\capsules\sel03.log" -Encoding utf8
Note "sel03 done (exit $LASTEXITCODE)"

Note "block-size cuts of sel03's first 20 trays: 1, 4, 16, whole tray"
& .\run.ps1 scripts/capsule_block_curve.py --log output/capsules/sel03 --sizes "1,4,16,0" --max-scans 20 *>&1 | Out-File -Append "output\capsules\sel03_curve.log" -Encoding utf8
Note "cuts done (exit $LASTEXITCODE)"

$logs = "output/capsules/sel03,output/capsules/sel03/blocks/n1,output/capsules/sel03/blocks/n4,output/capsules/sel03/blocks/n16,output/capsules/sel03/blocks/nall"
$arms = "d3-27b-vision,intern-4b-vision,qwen-27b-vision,qwen-27b-self-vision"
Note "replays: $arms"
& .\run.ps1 scripts/replay_capsules.py --logs $logs --arms $arms *>&1 | Out-File -Append "output\capsules\sel03_replay.log" -Encoding utf8
Note "replays done (exit $LASTEXITCODE)"

Note "catch rates"
& .\run.ps1 python scripts/capsule_catch_rates.py output/capsules/sel03 output/capsules/sel03/blocks/n1 output/capsules/sel03/blocks/n4 output/capsules/sel03/blocks/n16 output/capsules/sel03/blocks/nall *>&1 | Out-File -Append "output\capsules\sel03_catch_rates.log" -Encoding utf8
Note "all done (exit $LASTEXITCODE)"
