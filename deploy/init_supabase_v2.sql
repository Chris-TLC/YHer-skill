-- =============================================================
-- YHer-skill: Supabase 阶段 10 改造（多用户隔离）
-- =============================================================
-- 用法：在 Supabase Dashboard → SQL Editor 粘贴运行
-- 注意：此脚本只更新 RLS policy + 加索引，不删除任何数据
-- chris_2026 的历史记录会被完整保留
-- =============================================================

-- ─── 1. 删除阶段 8.5 的旧 policy（FOR ALL anon 全权限）───
DROP POLICY IF EXISTS "anon_all_users"            ON users;
DROP POLICY IF EXISTS "anon_all_user_profile"     ON user_profile;
DROP POLICY IF EXISTS "anon_all_query_history"    ON query_history;
DROP POLICY IF EXISTS "anon_all_memory_summaries" ON memory_summaries;

-- ─── 2. 新 policy：分操作类型（多用户友好）───
-- 策略：应用层强制 .eq('user_id', xxx)，policy 仅做基础保护

-- users 表
CREATE POLICY "anon_select_users" ON users
    FOR SELECT TO anon USING (true);
CREATE POLICY "anon_insert_users" ON users
    FOR INSERT TO anon WITH CHECK (true);
CREATE POLICY "anon_update_users" ON users
    FOR UPDATE TO anon USING (true);

-- user_profile 表
CREATE POLICY "anon_select_profile" ON user_profile
    FOR SELECT TO anon USING (true);
CREATE POLICY "anon_insert_profile" ON user_profile
    FOR INSERT TO anon WITH CHECK (true);
CREATE POLICY "anon_update_profile" ON user_profile
    FOR UPDATE TO anon USING (true);

-- query_history 表（FOR ALL 简化）
CREATE POLICY "anon_all_history" ON query_history
    FOR ALL TO anon USING (true) WITH CHECK (true);

-- memory_summaries 表（FOR ALL 简化）
CREATE POLICY "anon_all_summaries" ON memory_summaries
    FOR ALL TO anon USING (true) WITH CHECK (true);

-- ─── 3. 性能索引（多用户后按 user_id 大量查询）───
CREATE INDEX IF NOT EXISTS idx_users_user_id          ON users(user_id);
CREATE INDEX IF NOT EXISTS idx_profile_user_id        ON user_profile(user_id);
CREATE INDEX IF NOT EXISTS idx_history_user_created   ON query_history(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_user_period  ON memory_summaries(user_id, period);

-- ─── 4. 监控视图（方便看用户量和成本）───
CREATE OR REPLACE VIEW v_user_stats AS
SELECT
    COUNT(DISTINCT user_id) AS total_users,
    COUNT(*) AS total_queries,
    COALESCE(SUM(cost_yuan), 0) AS total_cost_yuan,
    DATE_TRUNC('day', created_at) AS date
FROM query_history
GROUP BY DATE_TRUNC('day', created_at)
ORDER BY date DESC;

-- ─── 5. 验证 ───
SELECT 'Tables' AS step, COUNT(*) AS count
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('users', 'user_profile', 'query_history', 'memory_summaries')

UNION ALL

SELECT 'Policies OK', COUNT(*)
FROM pg_policies
WHERE schemaname = 'public'
  AND tablename IN ('users', 'user_profile', 'query_history', 'memory_summaries')

UNION ALL

SELECT 'Indexes OK', COUNT(*)
FROM pg_indexes
WHERE schemaname = 'public'
  AND indexname LIKE 'idx_%';

-- 预期输出（更新后）:
--   Tables      : 4
--   Policies OK : 9 (3 users + 3 user_profile + 1 query_history + 1 memory_summaries + 1 可能旧policy)
--   Indexes OK  : ≥ 5
