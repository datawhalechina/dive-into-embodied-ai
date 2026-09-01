#!/usr/bin/env bash
# Fetch the Go2 visual meshes from the pinned unitree_rl_mjlab commit.
# The MJCF files reference them, so the model cannot load until they exist.
# Idempotent: files already present with the right checksum are skipped.
set -euo pipefail

COMMIT="1425b15f73bd4095f0df53709d7c389c3eb9e790"
BASE="https://raw.githubusercontent.com/unitreerobotics/unitree_rl_mjlab/$COMMIT"
UPSTREAM_DIR="src/assets/robots/unitree_go2/xmls/assets"
DEST="$(cd "$(dirname "$0")/.." && pwd)/src/unitree_rl_mjx/assets/robots/unitree_go2/xmls/assets"

CHECKSUMS="
8cdc53719002a5ee737d867904e534a6d6fa930b616bdcc1b3131e577d4e46c0  base_0.obj
39d2f4724ff64dd5a0640c308fd98902a2946660a8d033ec8dbf368fd5af0dd6  base_1.obj
29cae463f95d678e4ede863c36c9b19b0d356b8340412add10670d70bdcc3095  base_2.obj
f51ba43bf300e68957a0c1aa83eaf3e4926f4ac7e64a0d6c33ae5d21fcbcf5f6  base_3.obj
49ddec0129546e5dd2075d5beacf7f207baa2860aeff7626818e64d2efbf3821  base_4.obj
4a818cd563c5b7e2a271504442d05fd777d38cbaa028d4a10ad29c9f87bb355e  calf_0.obj
51950c3a153ae3208592df86ea11c9d6756f8f645ebca5caf6910c61aeb03817  calf_1.obj
6dc0406207a1c08a5953fd5962aca1a270fd06593d30d100205e05c1b32d4af2  calf_mirror_0.obj
f84f5af90b914633165beb3a5493ee499169144c5797dcf04c6eb5842f74b31a  calf_mirror_1.obj
df9e78a7c0110d02557439010f2a5f11f0ec4fd907b32704145dbd6aa7affc0b  foot.obj
cc29932ab4d251ce3f72389ec4c1f757e44eee138b87e772b648c2bc76b506cd  hip_0.obj
3fd5bd9b2ef6abe622c0739d3c67eaa1679b19b47132951972c4892a779ea7cf  hip_1.obj
bb3edfe6a7b04f4f4dff1ce7b05d426ddce941e276cc6372fb542055893c71ec  thigh_0.obj
3a631822f9f7d2114b1b79dcdff5abf95103630abb7d91d1f3b8b18117132ed2  thigh_1.obj
07abaedce761f1dd48647384ac14ba1959276bc78d5a2a62051e8f1c6093933b  thigh_mirror_0.obj
db984a3e983eabbc9b38183427cca9ab0bb3bdb049f9eab030e06edda36579d6  thigh_mirror_1.obj
"

mkdir -p "$DEST"
fetched=0 skipped=0
while read -r sum name; do
  [ -n "$name" ] || continue
  if [ -f "$DEST/$name" ] && echo "$sum  $DEST/$name" | shasum -a 256 -c - >/dev/null 2>&1; then
    skipped=$((skipped + 1))
    continue
  fi
  echo "fetching $name"
  curl -sfL "$BASE/$UPSTREAM_DIR/$name" -o "$DEST/$name"
  echo "$sum  $DEST/$name" | shasum -a 256 -c - >/dev/null
  fetched=$((fetched + 1))
done <<< "$CHECKSUMS"
echo "==> meshes ready in $DEST ($fetched fetched, $skipped already present)"
