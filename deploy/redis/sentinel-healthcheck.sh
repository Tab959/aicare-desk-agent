#!/bin/sh

# 使用受限客户端ACL验证Sentinel可返回当前master。
set -eu
export REDISCLI_AUTH="$(cat /run/secrets/agent-redis-sentinel-client-password)"
result="$(redis-cli --no-auth-warning -h 127.0.0.1 -p "${AICARE_SENTINEL_PORT:?AICARE_SENTINEL_PORT未配置}" --user aicare_sentinel SENTINEL get-master-addr-by-name aicare-agent-master)"
unset REDISCLI_AUTH
test -n "${result}"
