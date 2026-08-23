#!/bin/sh

# 生成独立Sentinel控制面ACL，并初始化可由Sentinel安全改写的持久化配置。
set -eu

client_secret_file="/run/secrets/agent-redis-sentinel-client-password"
peer_secret_file="/run/secrets/agent-redis-sentinel-peer-password"
management_secret_file="/run/secrets/agent-redis-sentinel-management-password"
runtime_dir="/run/aicare-sentinel"
acl_file="${runtime_dir}/users.acl"
config_file="/data/sentinel.conf"
sentinel_port="${AICARE_SENTINEL_PORT:?AICARE_SENTINEL_PORT未配置}"
master_name="aicare-agent-master"

for secret_file in "${client_secret_file}" "${peer_secret_file}" "${management_secret_file}"; do
    if [ ! -s "${secret_file}" ]; then
        echo "Agent Redis Sentinel启动失败：所需secret不存在或为空" >&2
        exit 1
    fi
done

install -d -m 0700 "${runtime_dir}"
client_password="$(cat "${client_secret_file}")"
peer_password="$(cat "${peer_secret_file}")"
management_password="$(cat "${management_secret_file}")"
client_hash="$(printf '%s' "${client_password}" | sha256sum | awk '{print $1}')"
peer_hash="$(printf '%s' "${peer_password}" | sha256sum | awk '{print $1}')"

printf '%s\n' \
    'user default off' \
    "user aicare_sentinel on #${client_hash} resetchannels -@all +auth +client|getname +client|id +client|setname +command +hello +ping +role +sentinel|get-master-addr-by-name +sentinel|master +sentinel|masters +sentinel|sentinels +sentinel|replicas" \
    "user aicare_sentinel_peer on #${peer_hash} allchannels +@all" \
    > "${acl_file}"
chmod 0600 "${acl_file}"

if [ ! -s "${config_file}" ]; then
    cat > "${config_file}" <<EOF
bind 0.0.0.0
protected-mode yes
port ${sentinel_port}
dir /data
aclfile ${acl_file}
sentinel monitor ${master_name} 192.168.150.105 6380 2
sentinel down-after-milliseconds ${master_name} 5000
sentinel failover-timeout ${master_name} 30000
sentinel parallel-syncs ${master_name} 1
sentinel auth-user ${master_name} aicare_sentinel_management
sentinel auth-pass ${master_name} ${management_password}
sentinel sentinel-user aicare_sentinel_peer
sentinel sentinel-pass ${peer_password}
EOF
    chmod 0600 "${config_file}"
fi

unset client_password peer_password management_password
exec redis-server "${config_file}" --sentinel
