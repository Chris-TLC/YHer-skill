-- ============================================================
-- YHer-skill: Supabase 数据库初始化脚本
-- ============================================================
-- 用法：在 Supabase Dashboard → SQL Editor 里整段粘贴运行
-- 创建时间：2026-05-06
-- ============================================================

-- ─── 1. 清理旧表（重新部署时使用，首次部署可跳过）───
DROP TABLE IF EXISTS memory_summaries CASCADE;
DROP TABLE IF EXISTS query_history CASCADE;
DROP TABLE IF EXISTS user_profile CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- ─── 2. 用户基本信息 ───
CREATE TABLE users (
    user_id     TEXT PRIMARY KEY,
    grade       TEXT,
    school      TEXT,
    region      TEXT,
    created_at  TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP DEFAULT NOW()
);

-- ─── 3. 用户档案（弱点、已掌握）───
CREATE TABLE user_profile (
    user_id          TEXT REFERENCES users(user_id),
    weak_topics      JSONB,
    mastered_topics  JSONB,
    learning_goals   TEXT,
    updated_at       TIMESTAMP DEFAULT NOW()
);

-- ─── 4. 提问历史（30 天后会被压缩到 memory_summaries）───
CREATE TABLE query_history (
    id                  SERIAL PRIMARY KEY,
    user_id             TEXT REFERENCES users(user_id),
    query               TEXT,
    diagnosis           JSONB,
    diagnosis_summary   TEXT,
    response_summary    TEXT,
    weak_topics_added   JSONB,
    cost_yuan           FLOAT,
    created_at          TIMESTAMP DEFAULT NOW()
);

-- ─── 5. 季度摘要（4000 tokens 高保真档案）───
CREATE TABLE memory_summaries (
    id                 SERIAL PRIMARY KEY,
    user_id            TEXT REFERENCES users(user_id),
    period             TEXT,             -- "2026-Q1"
    summary            TEXT,             -- 3500-4000 字精细档案
    compressed_count   INT,
    compression_ratio  TEXT,
    created_at         TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, period)
);

CREATE INDEX idx_summaries_user_period ON memory_summaries(user_id, period);
CREATE INDEX idx_history_user_date     ON query_history(user_id, created_at);

-- ─── 6. RLS Policy（保留 RLS + 为 anon 角色开全权限）───
-- 单用户场景或本地部署用此方案；多用户产品化时改为按 user_id 隔离
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_profile ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory_summaries ENABLE ROW LEVEL SECURITY;

CREATE POLICY "anon_all_users"             ON users             FOR ALL TO anon USING (true) WITH CHECK (true);
CREATE POLICY "anon_all_user_profile"      ON user_profile      FOR ALL TO anon USING (true) WITH CHECK (true);
CREATE POLICY "anon_all_query_history"     ON query_history     FOR ALL TO anon USING (true) WITH CHECK (true);
CREATE POLICY "anon_all_memory_summaries"  ON memory_summaries  FOR ALL TO anon USING (true) WITH CHECK (true);

-- ─── 7. 初始化默认用户（按需修改 user_id）───
INSERT INTO users (user_id, grade, school, region) 
VALUES ('default_user', '高二', '某重点中学', '上海');

INSERT INTO user_profile (user_id, weak_topics, mastered_topics)
VALUES ('default_user', '[]'::jsonb, '[]'::jsonb);

-- ─── 8. 验证：4 张表 + 4 个 policy ───
SELECT 'Tables created:' AS step, COUNT(*) AS count
FROM information_schema.tables 
WHERE table_schema = 'public' 
  AND table_name IN ('users', 'user_profile', 'query_history', 'memory_summaries')

UNION ALL

SELECT 'Policies created:', COUNT(*)
FROM pg_policies
WHERE schemaname = 'public'
  AND tablename IN ('users', 'user_profile', 'query_history', 'memory_summaries');

-- 预期输出：
--   Tables created:   4
--   Policies created: 4
