# Fetches the Whistle speech model and its Windows runner for spoken operator answers (src/handoff/voice.py).
# Pinned to exact Hugging Face commits and checked by hash. About 18.5 MB. The demo works without them: buttons only.
$ErrorActionPreference = "Stop"
$dest = "D:\Models\whistle"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$files = @(
  @{ url = "https://huggingface.co/Cactus-Compute/needle3/resolve/d6c7d6a2e9655505e0f8e23339356dd1705900eb/windows-x86_64/needle.exe"; name = "needle.exe"; sha = "0d4bc1da8c51f5600f5c4f4bb4a92e0abe142fae7b69e695ccab93e6745c883d" },
  @{ url = "https://huggingface.co/Cactus-Compute/whistle/resolve/741118687e1f008f06d1351a402a731274789790/whistle.cact"; name = "whistle.cact"; sha = "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb" }
)
foreach ($f in $files) {
  $path = Join-Path $dest $f.name
  if ((Test-Path $path) -and (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower() -eq $f.sha) {
    "$($f.name): already here"
    continue
  }
  Invoke-WebRequest -UseBasicParsing -Uri $f.url -OutFile $path
  $got = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
  if ($got -ne $f.sha) { Remove-Item $path; throw "$($f.name): hash $got, expected $($f.sha)" }
  "$($f.name): ok"
}
