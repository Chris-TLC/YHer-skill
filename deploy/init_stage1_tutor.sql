-- Stage 1 AI 私教学习舱数据结构
-- 用途：把一次单科托管学习沉淀为可追踪的学生画像、任务执行和能力变化。
--
-- 运行位置：Supabase SQL Editor
-- 依赖：已有 users / user_profile 表时可共存；没有也可单独使用。

create table if not exists tutor_student_profiles (
    id uuid primary key default gen_random_uuid(),
    user_id text not null,
    subject text not null default 'chemistry',
    grade text,
    region text,
    exam_system text,
    current_score numeric,
    target_score numeric,
    school_level text,
    learning_preference text,
    known_weaknesses jsonb not null default '[]'::jsonb,
    ability_scores jsonb not null default '{}'::jsonb,
    mastered_topics jsonb not null default '[]'::jsonb,
    weak_topics jsonb not null default '[]'::jsonb,
    updated_at timestamptz not null default now(),
    created_at timestamptz not null default now(),
    unique (user_id, subject)
);

create table if not exists tutor_sessions (
    id text primary key,
    user_id text not null,
    subject text not null default 'chemistry',
    goal text not null,
    topic_id text,
    topic_name text,
    time_budget_min int,
    status text not null default 'active',
    profile_snapshot jsonb not null default '{}'::jsonb,
    diagnosis_snapshot jsonb not null default '{}'::jsonb,
    ability_hypothesis jsonb not null default '{}'::jsonb,
    diagnostic_questions jsonb not null default '[]'::jsonb,
    task_queue jsonb not null default '[]'::jsonb,
    recommended_videos jsonb not null default '[]'::jsonb,
    final_report text,
    created_at timestamptz not null default now(),
    finished_at timestamptz
);

create table if not exists tutor_events (
    id bigserial primary key,
    session_id text not null references tutor_sessions(id) on delete cascade,
    user_id text not null,
    role text not null check (role in ('student', 'tutor', 'system')),
    task_id text,
    event_type text not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    cost_yuan numeric not null default 0,
    created_at timestamptz not null default now()
);

create table if not exists tutor_ability_snapshots (
    id bigserial primary key,
    session_id text not null references tutor_sessions(id) on delete cascade,
    user_id text not null,
    subject text not null default 'chemistry',
    source text not null,
    ability_scores jsonb not null default '{}'::jsonb,
    weak_axes jsonb not null default '[]'::jsonb,
    evidence jsonb not null default '[]'::jsonb,
    next_actions jsonb not null default '[]'::jsonb,
    created_at timestamptz not null default now()
);

create index if not exists idx_tutor_profiles_user_subject
    on tutor_student_profiles(user_id, subject);

create index if not exists idx_tutor_sessions_user_created
    on tutor_sessions(user_id, created_at desc);

create index if not exists idx_tutor_events_session_created
    on tutor_events(session_id, created_at);

create index if not exists idx_tutor_ability_user_created
    on tutor_ability_snapshots(user_id, subject, created_at desc);

alter table tutor_student_profiles enable row level security;
alter table tutor_sessions enable row level security;
alter table tutor_events enable row level security;
alter table tutor_ability_snapshots enable row level security;

-- Prototype policy:
-- 当前 Web BYOK Demo 仍使用 app 层 user_id 隔离；正式商业化前应改为 Supabase Auth uid。
drop policy if exists "prototype_profile_all" on tutor_student_profiles;
create policy "prototype_profile_all" on tutor_student_profiles
    for all using (true) with check (true);

drop policy if exists "prototype_sessions_all" on tutor_sessions;
create policy "prototype_sessions_all" on tutor_sessions
    for all using (true) with check (true);

drop policy if exists "prototype_events_all" on tutor_events;
create policy "prototype_events_all" on tutor_events
    for all using (true) with check (true);

drop policy if exists "prototype_ability_all" on tutor_ability_snapshots;
create policy "prototype_ability_all" on tutor_ability_snapshots
    for all using (true) with check (true);
