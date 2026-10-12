#!/bin/zsh
cd -- "${0:A:h:h}" || exit 1
/opt/anaconda3/envs/audio-ai/bin/python scripts/shengke_service.py status
read "?按回车关闭此窗口…"
