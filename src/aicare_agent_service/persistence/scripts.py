"""集中保存 Redis RunStore 使用的原子 Lua 脚本。

Redis在服务端一次性执行每段脚本，执行期间不会被其他命令穿插，因此可安全协调run hash、
conversation lease、active-run引用和cleanup guard。``KEYS[n]``是key参数，``ARGV[n]``
是普通值参数；Python调用处必须严格保持顺序。Lua注释会改变脚本缓存SHA但不改变执行语义。
"""

# 开始run脚本：处理清理guard、幂等摘要、恢复、单飞租约和首次安全记录创建。
BEGIN_RUN_LUA = r"""
-- Redis TIME返回服务器秒和微秒，统一封装成毫秒，避免依赖Python主机时钟。
local function now_ms()
    -- parts[1]为秒，parts[2]为微秒；Lua数组从1开始编号。
    local parts = redis.call('TIME')
    -- 秒乘1000，再把微秒除1000向下取整为毫秒。
    return tonumber(parts[1]) * 1000 + math.floor(tonumber(parts[2]) / 1000)
end

-- KEYS[4]是cleanup guard；存在时阻止任何新run和checkpoint删除并发。
if redis.call('GET', KEYS[4]) then
    -- 返回数组是为了与其他脚本返回结构保持一致。
    return {'IN_PROGRESS'}
end

-- KEYS[1]是run hash；request_digest存在表示该runId已有记录。
local existing_digest = redis.call('HGET', KEYS[1], 'request_digest')
if existing_digest then
    -- 相同runId但摘要不同属于请求冲突，不能覆盖旧记录。
    if existing_digest ~= ARGV[1] then
        return {'CONFLICT'}
    end

    -- 读取现有生命周期状态，决定重放、恢复或冲突。
    local status = redis.call('HGET', KEYS[1], 'status')
    -- 已完成run只允许Python从记录的checkpoint验证并重放终态。
    if status == 'COMPLETED' then
        return {'REPLAY_COMPLETED'}
    end
    -- FAILED/CANCELLED等其他终态不能重新执行同一runId。
    if status ~= 'RUNNING' then
        return {'CONFLICT'}
    end
    -- KEYS[2] conversation lease仍存在，说明已有当前执行者。
    if redis.call('GET', KEYS[2]) then
        return {'IN_PROGRESS'}
    end

    -- 旧lease已过期，使用新原始token以NX+PX原子尝试取得毫秒租约。
    local acquired = redis.call('SET', KEYS[2], ARGV[2], 'NX', 'PX', ARGV[4])
    -- 竞争失败表示另一个恢复者先取得租约。
    if not acquired then
        return {'IN_PROGRESS'}
    end

    -- 使用Redis服务器时间刷新记录中的租约诊断时间。
    local current_ms = now_ms()
    -- HSET一次更新租约摘要、理论过期时间和更新时间。
    redis.call('HSET', KEYS[1],
        'lease_token_digest', ARGV[3],
        'lease_expires_at_ms', current_ms + tonumber(ARGV[4]),
        'updated_at_ms', current_ms)
    -- 刷新run hash保留TTL，避免恢复期间记录过期。
    redis.call('EXPIRE', KEYS[1], ARGV[5])
    -- KEYS[3]保存哈希后的run key引用且RUNNING期间无TTL，保护checkpoint不被清理。
    redis.call('SET', KEYS[3], ARGV[12])
    -- 告诉Python使用ainvoke(None)从最新checkpoint恢复。
    return {'RESUMED'}
end

-- 新run前检查active-run引用；若目标run hash已过期则原子清除悬空引用。
local active_run_key = redis.call('GET', KEYS[3])
if active_run_key and redis.call('EXISTS', active_run_key) == 0 then
    redis.call('DEL', KEYS[3])
    -- Lua false表示后续不再把它视为活跃引用。
    active_run_key = false
end
-- 任一conversation lease或有效active-run存在都执行单飞阻断。
if redis.call('GET', KEYS[2]) or active_run_key then
    return {'IN_PROGRESS'}
end

-- 为全新run竞争取得conversation lease。
local acquired = redis.call('SET', KEYS[2], ARGV[2], 'NX', 'PX', ARGV[4])
if not acquired then
    return {'IN_PROGRESS'}
end

-- 记录首次开始的Redis服务器毫秒时间。
local current_ms = now_ms()
-- 创建只含Java身份、摘要、租约和时间的run hash，不保存任何对话正文。
redis.call('HSET', KEYS[1],
    'tenant_id', ARGV[6],
    'customer_id', ARGV[7],
    'conversation_id', ARGV[8],
    'run_id', ARGV[9],
    'trigger_message_id', ARGV[10],
    'trigger_sequence', ARGV[11],
    'request_digest', ARGV[1],
    'status', 'RUNNING',
    'lease_token_digest', ARGV[3],
    'lease_expires_at_ms', current_ms + tonumber(ARGV[4]),
    'started_at_ms', current_ms,
    'updated_at_ms', current_ms)
-- run记录按配置保留，终态提交时会再次刷新TTL。
redis.call('EXPIRE', KEYS[1], ARGV[5])
-- 设置conversation到哈希run key的活跃引用，终态脚本负责删除。
redis.call('SET', KEYS[3], ARGV[12])
-- 当前调用者拥有租约，可开始首次图执行。
return {'STARTED'}
"""


# 安全释放cleanup guard：只有值与调用者token相同才删除，避免误删他人的锁。
RELEASE_CLEANUP_GUARD_LUA = r"""
-- KEYS[1]是guard key，ARGV[1]是申请时返回的所有权token。
if redis.call('GET', KEYS[1]) == ARGV[1] then
    -- 比较与删除在同一脚本中原子完成；DEL返回实际删除数量。
    return redis.call('DEL', KEYS[1])
end
-- key不存在或token不匹配均返回0且不修改状态。
return 0
"""


# 租约续期脚本：验证RUNNING和原始token后同时刷新lease、记录时间及TTL。
RENEW_LEASE_LUA = r"""
-- KEYS[1]为run hash，先读取状态判断记录是否存在。
local status = redis.call('HGET', KEYS[1], 'status')
if not status then
    return {'NOT_FOUND'}
end
-- 终态run不能再续租。
if status ~= 'RUNNING' then
    return {'INVALID_TRANSITION'}
end
-- KEYS[2]为lease key，其值必须等于ARGV[1]原始token。
if redis.call('GET', KEYS[2]) ~= ARGV[1] then
    return {'LEASE_LOST'}
end

-- 续租时间统一取Redis服务器时钟。
local parts = redis.call('TIME')
local current_ms = tonumber(parts[1]) * 1000 + math.floor(tonumber(parts[2]) / 1000)
-- ARGV[2]是毫秒租约长度，PEXPIRE原子延长lease TTL。
redis.call('PEXPIRE', KEYS[2], ARGV[2])
-- 同步刷新hash中的诊断过期时间和更新时间。
redis.call('HSET', KEYS[1],
    'lease_expires_at_ms', current_ms + tonumber(ARGV[2]),
    'updated_at_ms', current_ms)
-- ARGV[3]是run记录秒级保留期。
redis.call('EXPIRE', KEYS[1], ARGV[3])
return {'RENEWED'}
"""


# 取消意图脚本：Java可请求停止RUNNING run，但不在此处越权提交CANCELLED。
REQUEST_CANCEL_LUA = r"""
-- 读取run状态并区分不存在和非法终态迁移。
local status = redis.call('HGET', KEYS[1], 'status')
if not status then
    return {'NOT_FOUND'}
end
if status ~= 'RUNNING' then
    return {'INVALID_TRANSITION'}
end
-- 已记录取消时保持幂等，不覆盖首次请求时间。
if redis.call('HEXISTS', KEYS[1], 'cancel_requested_at_ms') == 1 then
    return {'ALREADY_REQUESTED'}
end

-- 使用Redis服务器时间记录取消意图和最近更新时间。
local parts = redis.call('TIME')
local current_ms = tonumber(parts[1]) * 1000 + math.floor(tonumber(parts[2]) / 1000)
redis.call('HSET', KEYS[1],
    'cancel_requested_at_ms', current_ms,
    'updated_at_ms', current_ms)
-- ARGV[1]刷新run hash保留期，给当前执行者时间观察并提交取消。
redis.call('EXPIRE', KEYS[1], ARGV[1])
return {'CANCEL_REQUESTED'}
"""


# 终态脚本：验证状态与租约，幂等提交COMPLETED/FAILED/CANCELLED并释放单飞资源。
TERMINATE_RUN_LUA = r"""
-- KEYS[1]为run hash；不存在时不能创建凭空终态。
local status = redis.call('HGET', KEYS[1], 'status')
if not status then
    return {'NOT_FOUND'}
end

-- 已是目标终态时进一步核对语义，实现严格幂等而非盲目成功。
if status == ARGV[2] then
    -- COMPLETED必须checkpoint ID和final digest均完全一致。
    if ARGV[2] == 'COMPLETED' then
        if redis.call('HGET', KEYS[1], 'checkpoint_id') == ARGV[3]
            and redis.call('HGET', KEYS[1], 'final_digest') == ARGV[4] then
            return {'ALREADY_TERMINAL'}
        end
        return {'INVALID_TRANSITION'}
    end
    -- FAILED重复提交必须保持同一稳定错误码。
    if ARGV[2] == 'FAILED' and redis.call('HGET', KEYS[1], 'error_code') ~= ARGV[5] then
        return {'INVALID_TRANSITION'}
    end
    -- CANCELLED或语义一致的FAILED可安全视作已经终止。
    return {'ALREADY_TERMINAL'}
end

-- 不允许从一个终态转换到另一个终态。
if status ~= 'RUNNING' then
    return {'INVALID_TRANSITION'}
end
-- 只有当前conversation lease原始token持有者可提交终态。
if redis.call('GET', KEYS[2]) ~= ARGV[1] then
    return {'LEASE_LOST'}
end

-- 终态时间使用Redis服务器时钟。
local parts = redis.call('TIME')
local current_ms = tonumber(parts[1]) * 1000 + math.floor(tonumber(parts[2]) / 1000)
-- 原子写目标状态、更新时间和完成时间。
redis.call('HSET', KEYS[1],
    'status', ARGV[2],
    'updated_at_ms', current_ms,
    'completed_at_ms', current_ms)
-- 终态后删除租约摘要、过期时间和临时取消意图。
redis.call('HDEL', KEYS[1],
    'lease_token_digest',
    'lease_expires_at_ms',
    'cancel_requested_at_ms')

-- COMPLETED保存checkpoint标识及去eventIndex终态摘要，不保存回答正文。
if ARGV[2] == 'COMPLETED' then
    redis.call('HSET', KEYS[1], 'checkpoint_id', ARGV[3], 'final_digest', ARGV[4])
-- FAILED只保存稳定错误码。
elseif ARGV[2] == 'FAILED' then
    redis.call('HSET', KEYS[1], 'error_code', ARGV[5])
end

-- 释放conversation lease。
redis.call('DEL', KEYS[2])
-- 仅当active-run仍指向当前run key时删除，避免误删新执行引用。
if redis.call('GET', KEYS[3]) == ARGV[7] then
    redis.call('DEL', KEYS[3])
end
-- 终态记录继续按ARGV[6]秒保留，支持幂等重试或完成重放。
redis.call('EXPIRE', KEYS[1], ARGV[6])
return {'TERMINATED'}
"""


# 活跃检查脚本：供checkpoint清理器无正文判断run/lease，并自愈悬空active引用。
CHECK_CONVERSATION_ACTIVE_LUA = r"""
-- KEYS[1]为conversation active-run引用key。
local active_run_key = redis.call('GET', KEYS[1])
if active_run_key then
    -- 引用的run hash仍存在，说明RUNNING状态仍需保护checkpoint。
    if redis.call('EXISTS', active_run_key) == 1 then
        return 1
    end
    -- run hash已过期则删除悬空引用，避免永久阻止清理。
    redis.call('DEL', KEYS[1])
end
-- KEYS[2]有效lease存在也代表会话正在执行。
if redis.call('GET', KEYS[2]) then
    return 1
end
-- 无run引用且无lease，返回非活跃。
return 0
"""


# 获取cleanup guard脚本：再次原子检查guard、active-run和lease后以NX+EX加锁。
ACQUIRE_CLEANUP_GUARD_LUA = r"""
-- KEYS[3]已有guard表示另一个清理器正在处理同一会话。
if redis.call('GET', KEYS[3]) then
    return 0
end
-- 检查active-run引用，并与活跃检查脚本一样自愈悬空引用。
local active_run_key = redis.call('GET', KEYS[1])
if active_run_key then
    if redis.call('EXISTS', active_run_key) == 1 then
        return 0
    end
    redis.call('DEL', KEYS[1])
end
-- 有效lease存在时禁止开始清理。
if redis.call('GET', KEYS[2]) then
    return 0
end
-- ARGV[1]是随机所有权token，ARGV[2]是秒级guard TTL；NX保证竞争唯一获胜者。
local acquired = redis.call('SET', KEYS[3], ARGV[1], 'NX', 'EX', ARGV[2])
if acquired then
    return 1
end
-- 竞争失败返回0，调用者按活跃/不可清理处理。
return 0
"""
