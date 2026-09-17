# 启动视频改写剧本工作台（带访问口令，适合对外暴露）
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:VSS_ACCESS_TOKEN = (Get-Content "$PSScriptRoot\.access-token.txt" -Raw).Trim()
$env:PORT = "8790"
# 想用云端大模型就取消下面三行的注释并填入
# $env:VSS_LLM_BASE_URL = "https://api.deepseek.com/v1"
# $env:VSS_LLM_API_KEY  = "你的key"
# $env:VSS_LLM_CLOUD_MODEL = "deepseek-chat"
Write-Host "启动中… 口令见 .access-token.txt"
python app.py
