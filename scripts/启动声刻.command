#!/bin/zsh
cd -- "${0:A:h:h}" || exit 1
export PATH="/Users/$USER/.docker/bin:/opt/anaconda3/bin:$PATH"
/opt/anaconda3/envs/audio-ai/bin/python scripts/shengke_service.py start --open
result=$?
if (( result != 0 )); then
  printf '\n启动未完成，请查看上方提示。按回车关闭。\n'
  read -r
fi
exit $result
