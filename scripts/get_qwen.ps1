# Fetches Qwen3.8-27B (the vision-language model d3 27B is built on) as a 4-bit GGUF with its vision projector, for the
# same-backbone VDM vs VLM comparison (dup, Nov 4). IQ4_XS to match d3's own IQ4_XS file in size and precision.
# Pinned to an exact Hugging Face commit, checked by sha256, resumable. About 15.2 GB. Apache-2.0.
$ErrorActionPreference = "Stop"
$dest = "D:\Models\gguf\qwen3.8-27b"
$rev = "4ca720788d1e01f1bff70c033e0d0028fd02e502"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$files = @(
  @{ name = "Qwen3.8-27B-UD-IQ4_XS.gguf"; sha = "40fac4050e940397dbf13087afd50f4734a11805bf9d65ef8ddd7483470e6199" },
  @{ name = "mmproj-BF16.gguf"; sha = "83ee4f4f205fa514161778c41df1ea14144faa0f713510893b63c2395f5c2d53" }
)
foreach ($f in $files) {
  $path = Join-Path $dest $f.name
  if ((Test-Path $path) -and (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower() -eq $f.sha) {
    "$($f.name): already here"
    continue
  }
  $url = "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/$rev/$($f.name)"
  & curl.exe -L --fail --retry 5 -C - -o $path $url
  if ($LASTEXITCODE -ne 0) { throw "$($f.name): curl exited $LASTEXITCODE" }
  $got = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLower()
  if ($got -ne $f.sha) { Remove-Item $path; throw "$($f.name): hash $got, expected $($f.sha)" }
  "$($f.name): ok"
}
