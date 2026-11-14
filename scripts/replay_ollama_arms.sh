#!/usr/bin/env bash
# Usage: [LOGS="output/sort/test03"] bash scripts/replay_ollama_arms.sh [arm ...]   (from WSL; arms must be in TAG below)
# One Ollama decision arm at a time on test03 then log02b, 50 answers per chunk, model unloaded after each chunk,
# stop if Windows commit passes 110 GB. Resumable: replay skips answered decisions.
cd /mnt/c/Users/dup/dev.win/GDG-Hackathon
NATIVE=http://127.0.0.1:11435
declare -A TAG=([tev1-0.8b-text]=tev1:0.8b [d1-3b-vision]=d1:3b [tev1-4b-text]=tev1:4b [nimble-v3-text]=nimble-v3:latest [clef-flash-vision]=clef-flash:latest [d1-omni-vision]=d1-omni)
commit_gb() { powershell.exe -NoProfile -Command '$o=Get-CimInstance Win32_OperatingSystem; [math]::Round(($o.TotalVirtualMemorySize-$o.FreeVirtualMemory)/1MB,1)' | tr -d '\r'; }
unload() { curl -s "$NATIVE/api/generate" -d "{\"model\":\"$1\",\"keep_alive\":0}" >/dev/null; }
ARMS="${*:-tev1-0.8b-text d1-3b-vision tev1-4b-text nimble-v3-text clef-flash-vision}"
for arm in $ARMS; do
  for log in ${LOGS:-output/sort/test03 output/sort/log02b}; do
    while true; do
      c=$(commit_gb); echo "$(TZ=America/Toronto date +%I:%M%p) $arm $log commit ${c} GB"
      if awk "BEGIN{exit !($c > 110)}"; then echo "STOP: commit ${c} GB over 110"; unload "${TAG[$arm]}"; exit 1; fi
      out=$(timeout 1800 powershell.exe -NoProfile -ExecutionPolicy Bypass -File run.ps1 python -u scripts/benchmark.py replay --log $log --arm $arm --url $NATIVE --limit 50 2>&1 | tr -d '\r')
      unload "${TAG[$arm]}"
      echo "$out" | grep -E "new answers|Error|Traceback" | tail -2
      new=$(echo "$out" | grep -oE "^[0-9]+ new answers" | grep -oE "^[0-9]+")
      errs=$(echo "$out" | grep -cE "^XX .*(Error|error)")
      [ -z "$new" ] && { echo "replay failed for $arm"; echo "$out" | tail -5; break; }
      [ "$new" -eq 0 ] && break
      [ "$errs" -ge "$new" ] && { echo "every answer errored for $arm; skipping"; break; }
    done
  done
done
echo "ALL OLLAMA ARMS DONE"
