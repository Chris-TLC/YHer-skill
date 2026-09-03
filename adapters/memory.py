#!/usr/bin/env python3
"""
Yihuier AI chemistry assistant - long-term memory module (Supabase)
v3: full always-on memory + quarterly high-fidelity compression (4000 tokens)
"""

from typing import List, Dict
from datetime import datetime, timedelta
from collections import defaultdict


# ═══════════════════════════════════════════════════
# High-fidelity compression prompt (key to v3)
# ═══════════════════════════════════════════════════

HIGH_FIDELITY_COMPRESSION_PROMPT = """
你是一化儿教学体系的学习历史精细压缩师。

任务：把用户一个季度（约 90 天）的提问历史压缩成 3500-4000 字的【高保真】学习档案。

## 输出结构（6 部分，按顺序）

### 一、弱点演化轨迹（800 字）
- 季度内暴露的所有根因诊断（按出现频次排序，列前 10）
- 每个弱点的具体表现（保留 1-2 个原始问题作证据）
- 弱点之间的因果链
- 弱点修复进度

### 二、知识掌握进展（800 字）
- 季度内新掌握的知识点（含掌握的判定证据）
- 仍需巩固的知识点（按知识图谱节点列出）
- 知识图谱节点的"已访问/已掌握"状态变化
- 触及的章节：必修一/二/选修三/选修四/选修五

### 三、题型应对能力（700 字）
- 季度内遇到的所有题型（按 exam_pattern_id 列出，含频次）
- 每种题型的解题熟练度变化
- 高频犯错的题型 + 错点模式
- 题型变体：基础版掌握情况 vs 拔高版

### 四、学习行为模式（500 字）
- 提问时段分布（早/中/晚/深夜）
- 提问类型偏好（概念查询 / 解题求助 / 综合诊断）
- 学习节奏（密集学习日 vs 间歇日）
- 与考试时间表的关联

### 五、教师建议链（700 字）
- 杰哥本季度给的核心建议（按时间排序）
- 推荐过的视频学习路径（保留 BV+P 号）
- 思维招式使用情况
- 下个季度重点突破方向

### 六、量化指标（300 字）
- 本季度提问数: N
- 涉及知识节点数: N（其中新增 N 个）
- 涉及题型数: N
- 主要思维招式: [...]
- 与上季度对比的进步指标
- 累计弱点数 / 累计已掌握数

## 输出格式要求

- 纯文本，按上述 6 个标题组织
- 保留具体细节（如具体题目原文片段、BV+P 号）
- 不要泛化为"用户主要在学化学"这种无信息废话
- 不要 markdown 装饰，保持纯文本可被任何 LLM 读取
"""


class YihuierMemory:
    """Supabase long-term memory (v3 full memory + quarterly high-fidelity compression)."""

    def __init__(self, supabase_url: str, supabase_key: str):
        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_URL 和 SUPABASE_KEY 不能为空。"
                "请设置环境变量或传入参数。"
            )
        from supabase import create_client
        self.supabase = create_client(supabase_url, supabase_key)

    # ========== User profile ========================

    def get_user_profile(self, user_id: str) -> dict:
        """Fetch the user profile (merges users + user_profile across tables, isolated by user_id)."""
        user_data = {}
        profile_data = {}

        try:
            user_result = self.supabase.table('users')\
                .select('*').eq('user_id', user_id).execute()
            if user_result.data:
                user_data = user_result.data[0]
        except Exception as e:
            print(f"[memory] querying users table failed: {e}")

        try:
            profile_result = self.supabase.table('user_profile')\
                .select('*').eq('user_id', user_id).execute()
            if profile_result.data:
                profile_data = profile_result.data[0]
        except Exception as e:
            print(f"[memory] querying user_profile table failed: {e}")

        return {
            'user_id': user_id,
            'grade': user_data.get('grade'),
            'school': user_data.get('school'),
            'region': user_data.get('region'),
            'weak_topics': profile_data.get('weak_topics') or [],
            'mastered_topics': profile_data.get('mastered_topics') or [],
            'learning_goals': profile_data.get('learning_goals', ''),
        }

    def sync_user_info(self, user_id: str, grade: str = None,
                       name: str = None, school: str = None):
        """Sync basic user info to Supabase (used by the web frontend)."""
        import traceback
        try:
            user_data = {'user_id': user_id}
            if grade:
                user_data['grade'] = grade
            if name:
                user_data['school'] = name
            if school:
                user_data['school'] = school

            self.supabase.table('users').upsert(user_data).execute()

            # Make sure a user_profile row exists
            self.supabase.table('user_profile').upsert({
                'user_id': user_id,
            }).execute()
        except Exception as e:
            print(f"[memory] ❌ sync_user_info failed: {e}")
            traceback.print_exc()

    def update_weak_topics(self, user_id: str, new_weak: list):
        """Incrementally update the weak-topic list from diagnosis results."""
        try:
            profile = self.get_user_profile(user_id)
            existing = set(profile.get('weak_topics') or [])
            existing.update(new_weak)

            # Deduplicate + cap the count
            merged = list(existing)[:20]

            self.supabase.table('user_profile').upsert({
                'user_id': user_id,
                'weak_topics': merged,
            }).execute()
        except Exception as e:
            print(f"[memory] ❌ update_weak_topics failed: {e}")

    # ========== Cache-friendly segmented memory (v3.1) =

    def get_static_memory_section(self, user_id: str) -> str:
        """
        v3.1 static memory: USER_PROFILE + HISTORICAL_SUMMARIES
        Fetched once per session start and appended to the system_prompt.
        Stable for 24h → high prompt-caching hit rate.
        """
        try:
            # User profile
            profile = self.get_user_profile(user_id)
            weak_str = ', '.join(profile.get('weak_topics') or [])
            master_str = ', '.join(profile.get('mastered_topics') or [])

            # Quarterly summaries
            summaries = self.supabase.table('memory_summaries')\
                .select('*').eq('user_id', user_id)\
                .order('period').execute()

            if summaries.data:
                summary_text = "\n\n".join([
                    f"### {s['period']} 学习档案\n{s['summary']}"
                    for s in summaries.data
                ])
            else:
                summary_text = "（首次使用，暂无历史档案）"

            return f"""[USER_PROFILE]
grade: {profile.get('grade') or '未设置'}
weak_topics: [{weak_str}]
mastered_topics: [{master_str}]

[HISTORICAL_SUMMARIES（季度高保真档案）]
{summary_text}"""

        except Exception:
            return "[USER_PROFILE]\n（记忆系统暂不可用）"

    def get_dynamic_memory_section(self, user_id: str) -> str:
        """
        v3.1 dynamic memory: RECENT_30_DAYS_HISTORY
        May change with every question → goes into the user_message,
        so it doesn't break system_prompt cache hits.
        """
        try:
            cutoff = (datetime.now() - timedelta(days=30)).isoformat()
            recent = self.supabase.table('query_history')\
                .select('*').eq('user_id', user_id)\
                .gte('created_at', cutoff)\
                .order('created_at', desc=True).execute()

            if recent.data:
                recent_text = "\n".join([
                    f"- [{q['created_at'][:10]}] 问: {q['query'][:80]} → "
                    f"诊断: {(q.get('diagnosis_summary') or '')[:100]}"
                    for q in recent.data
                ])
            else:
                recent_text = "（最近 30 天无记录）"

            return f"[RECENT_30_DAYS_HISTORY]\n{recent_text}"
        except Exception:
            return "[RECENT_30_DAYS_HISTORY]\n（暂时不可用）"

    def get_full_memory_with_summaries(self, user_id: str) -> str:
        """
        Legacy-interface compatibility: return the full memory (static + dynamic).
        New code should use get_static_memory_section + get_dynamic_memory_section.
        """
        return (self.get_static_memory_section(user_id) + "\n\n" +
                self.get_dynamic_memory_section(user_id))

    # ========== History writes ======================

    def save_query(self, user_id: str, query: str, diagnosis: dict,
                   response: str, cost: float):
        """Save a query to history (errors are raised explicitly; silent failure is an engineering-discipline bug)."""
        import traceback
        try:
            diag_summary = self._summarize_diagnosis(diagnosis)

            self.supabase.table('query_history').insert({
                'user_id': user_id,
                'query': query,
                'diagnosis': diagnosis,
                'diagnosis_summary': diag_summary,
                'response_summary': response[:200],
                'weak_topics_added': diagnosis.get('missing_prereqs', []),
                'cost_yuan': cost,
            }).execute()
        except Exception as e:
            print(f"[memory] ❌ save_query failed: {type(e).__name__}: {e}")
            traceback.print_exc()

    def _summarize_diagnosis(self, diagnosis: dict) -> str:
        """Condense diagnose.py's output into a one-liner."""
        nodes = ', '.join(diagnosis.get('related_nodes', [])[:3])
        prereqs = ', '.join(diagnosis.get('missing_prereqs', [])[:3])
        return f"涉及 [{nodes}]，缺漏 [{prereqs}]"

    # ========== Quarterly compression (key to v3) ===

    def get_compression_status(self, user_id: str) -> dict:
        """Check whether compression is needed (count of records older than 90 days)."""
        try:
            cutoff = (datetime.now() - timedelta(days=90)).isoformat()
            result = self.supabase.table('query_history')\
                .select('id', count='exact')\
                .eq('user_id', user_id)\
                .lt('created_at', cutoff).execute()

            return {
                'needs_compression': result.count > 30,
                'old_records': result.count,
            }
        except Exception as e:
            print(f"[memory] ❌ get_compression_status failed: {e}")
            return {'needs_compression': False, 'old_records': 0}

    def compress_old_memory(self, user_id: str, llm_client) -> dict:
        """
        v3 quarterly high-fidelity compression: history older than 90 days → a detailed ~4000-token summary.

        Cost: once per quarter × ~4000 tokens × ¥6.26/M = ¥0.025
        24-month total: ~8 runs × ¥0.025 = ¥0.20
        """
        try:
            cutoff = datetime.now() - timedelta(days=90)
            old_records = self.supabase.table('query_history')\
                .select('*').eq('user_id', user_id)\
                .lt('created_at', cutoff.isoformat())\
                .execute()

            if len(old_records.data) < 30:
                return {'compressed': 0, 'cost': 0, 'periods': []}

            # Group by quarter
            by_quarter = defaultdict(list)
            for r in old_records.data:
                date_str = r['created_at'][:10]
                year, month, _ = date_str.split('-')
                quarter = (int(month) - 1) // 3 + 1
                period = f"{year}-Q{quarter}"
                by_quarter[period].append(r)

            total_cost = 0
            compressed_count = 0
            compressed_periods = []

            for period, records in by_quarter.items():
                # Check whether a summary already exists
                existing = self.supabase.table('memory_summaries')\
                    .select('*').eq('user_id', user_id)\
                    .eq('period', period).execute()

                if existing.data:
                    continue

                # Concatenate the raw records
                records_text = "\n".join([
                    f"[{r['created_at'][:10]}] 问: {r['query']} | "
                    f"诊断: {r.get('diagnosis_summary', '')} | "
                    f"新增弱点: {r.get('weak_topics_added', [])}"
                    for r in records
                ])

                # v3: high-fidelity compression with max_tokens=4000
                summary_response = llm_client.chat(
                    messages=[
                        {"role": "system",
                         "content": HIGH_FIDELITY_COMPRESSION_PROMPT},
                        {"role": "user", "content": records_text}
                    ],
                    max_tokens=4000,
                )

                # Store the summary
                self.supabase.table('memory_summaries').insert({
                    'user_id': user_id,
                    'period': period,
                    'summary': summary_response['content'],
                    'compressed_count': len(records),
                    'compression_ratio':
                        f"{len(records_text)//4}:{len(summary_response['content'])//4}",
                    'created_at': datetime.now().isoformat()
                }).execute()

                # Delete the raw records
                for r in records:
                    self.supabase.table('query_history')\
                        .delete().eq('id', r['id']).execute()

                total_cost += summary_response['cost_yuan']
                compressed_count += len(records)
                compressed_periods.append(period)

            return {
                'compressed': compressed_count,
                'cost': total_cost,
                'periods': compressed_periods,
            }
        except Exception as e:
            import traceback
            print(f"[memory] ❌ compress_old_memory failed: {e}")
            traceback.print_exc()
            return {'compressed': 0, 'cost': 0, 'periods': []}

    # ========== Cost queries ========================

    def get_month_cost(self, user_id: str, month: str) -> float:
        """Query the accumulated cost for a month; month format 'YYYY-MM'."""
        try:
            # Use the first day of the next month as the upper bound (avoid invalid dates like 01-32)
            year, m = month.split('-')
            y, mo = int(year), int(m)
            if mo == 12:
                next_month = f"{y+1}-01"
            else:
                next_month = f"{y}-{mo+1:02d}"

            result = self.supabase.table('query_history')\
                .select('cost_yuan')\
                .eq('user_id', user_id)\
                .gte('created_at', f'{month}-01')\
                .lt('created_at', f'{next_month}-01').execute()

            return sum(r['cost_yuan'] for r in result.data if r['cost_yuan'])
        except Exception as e:
            print(f"[memory] ❌ get_month_cost failed: {e}")
            return 0.0
