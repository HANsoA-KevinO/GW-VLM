#!/usr/bin/env bash
# 按 manifest.tsv 批量下载 arXiv PDF,校验 %PDF,记录成功/失败。
cd "$(dirname "$0")" || exit 1
: > fails.txt; : > ok.txt
n=0; ok=0; fail=0
while IFS=$'\t' read -r id topic slug; do
  [ -z "$id" ] && continue
  n=$((n+1))
  mkdir -p "$topic"
  out="$topic/${slug}__${id}.pdf"
  if [ -s "$out" ] && head -c4 "$out" | grep -q "%PDF"; then
    printf '%s\t%s\t%s\n' "$id" "$topic" "$slug" >> ok.txt; ok=$((ok+1)); echo "skip $out"; continue
  fi
  got=0
  for url in "https://arxiv.org/pdf/${id}" "https://arxiv.org/pdf/${id}v1" "https://arxiv.org/pdf/${id}v2"; do
    curl -sS -L --max-time 90 -o "$out" "$url" 2>/dev/null
    if head -c4 "$out" 2>/dev/null | grep -q "%PDF"; then got=1; break; fi
  done
  if [ "$got" = 1 ]; then
    printf '%s\t%s\t%s\n' "$id" "$topic" "$slug" >> ok.txt; ok=$((ok+1))
    echo "OK   $out ($(stat -f%z "$out")B)"
  else
    rm -f "$out"; printf '%s\t%s\t%s\n' "$id" "$topic" "$slug" >> fails.txt; fail=$((fail+1))
    echo "FAIL $id ($slug)"
  fi
done < manifest.tsv
echo "================ 汇总: 共$n 成功$ok 失败$fail ================"
