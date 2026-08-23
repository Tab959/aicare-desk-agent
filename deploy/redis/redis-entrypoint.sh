#!/bin/sh

# 初始化持久化Redis拓扑配置，并从只读secret生成运行时ACL与复制认证配置。
set -eu

app_secret_file="/run/secrets/agent-redis-password"
replication_secret_file="/run/secrets/agent-redis-replication-password"
management_secret_file="/run/secrets/agent-redis-sentinel-management-password"
runtime_dir="/run/aicare-redis"
acl_file="${runtime_dir}/users.acl"
auth_file="${runtime_dir}/auth.conf"
config_file="/data/redis.conf"
redis_port="${AICARE_REDIS_PORT:?AICARE_REDIS_PORT未配置}"

for secret_file in "${app_secret_file}" "${replication_secret_file}" "${management_secret_file}"; do
    if [ ! -s "${secret_file}" ]; then
        echo "Agent Redis启动失败：所需secret不存在或为空" >&2
        exit 1
    fi
done

install -d -m 0700 "${runtime_dir}"
app_password="$(cat "${app_secret_file}")"
replication_password="$(cat "${replication_secret_file}")"
management_password="$(cat "${management_secret_file}")"
app_hash="$(printf '%s' "${app_password}" | sha256sum | awk '{print $1}')"
replication_hash="$(printf '%s' "${replication_password}" | sha256sum | awk '{print $1}')"
management_hash="$(printf '%s' "${management_password}" | sha256sum | awk '{print $1}')"

printf '%s\n' \
    'user default off' \
    "user aicare_agent on #${app_hash} ~aicare:agent:{run-store}:* +ping +info +role +acl|whoami +config|get +script|load +eval +evalsha +get +set +del +exists +expire +pexpire +hget +hgetall +hset +hdel +hexists +time" \
    "user aicare_replication on #${replication_hash} ~* +ping +psync +replconf" \
    "user aicare_sentinel_management on #${management_hash} resetchannels &__sentinel__:hello +multi +slaveof +replicaof +ping +exec +subscribe +config|rewrite +role +publish +info +client|setname +client|kill +script|kill" \
    > "${acl_file}"
chmod 0600 "${acl_file}"

# 两个节点都预置复制凭据，被Sentinel降级为副本后可立即连接新master。
printf '%s\n' \
    "aclfile ${acl_file}" \
    'masteruser aicare_replication' \
    "masterauth ${replication_password}" \
    > "${auth_file}"
chmod 0600 "${auth_file}"

# 首次创建基础文件；之后Sentinel的CONFIG REWRITE会持久化真实角色。
if [ ! -s "${config_file}" ]; then
    {
        printf '%s\n' \
            'bind 0.0.0.0' \
            'protected-mode yes' \
            "port ${redis_port}" \
            'appendonly yes' \
            'appendfsync everysec' \
            'maxmemory-policy noeviction' \
            'save 900 1' \
            'save 300 10' \
            'save 60 10000' \
            'dir /data' \
            "include ${auth_file}"
        if [ "${AICARE_REDIS_ROLE:-primary}" = 'replica' ]; then
            printf '%s\n' 'replicaof 192.168.150.105 6380'
        fi
    } > "${config_file}"
    chmod 0600 "${config_file}"
fi

unset app_password replication_password management_password
exec redis-server "${config_file}"
