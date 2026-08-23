#!/bin/sh

# 使用应用ACL执行PING；密码经REDISCLI_AUTH传递，不出现在进程参数中。
set -eu
export REDISCLI_AUTH="$(cat /run/secrets/agent-redis-password)"
exec redis-cli --no-auth-warning -h 127.0.0.1 -p "${AICARE_REDIS_PORT:?AICARE_REDIS_PORT未配置}" --user aicare_agent ping
