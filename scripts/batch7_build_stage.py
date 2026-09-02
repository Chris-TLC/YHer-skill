#!/usr/bin/env python3
"""Batch 7 re-apply:生成 v4.1 暂存 manifest(方案 PROJECT_HANDOFF/BATCH7_REAPPLY_PLAN_2026-07-04.md)。

只写 /tmp/yher_batch7_stage/。官方一字不碰;门禁全过后由 apply 步骤拷入。
逐行处置(基底=官方 3329,零增零删):
  A pool→excluded_not_a_question : 答案202501 全组 19 + 考试须知 4(签字 subtype)
  B pool→excluded_bad_segmentation: 排除清单 9(静安7+嘉定2;答案202501 的 2 条归 A)
  C pool→excluded_answerless      : 签字 42 − 4(归A) − 2(答案202501重叠归A) = 36
  D 答案继承注入                   : 签字 7(v3 kg/rubric/standard_solution 深拷贝,manual_inherited)
  E 块修复交换                     : OMML 服务池 22(仅换 stem/analysis/answer 块+stem_text,保 id/对齐/池)
  F round2_042 对齐改判            : manual_no_split_keep_composite + KG 3 节点
  其余仅 schema_version 升 ws3_schema_v4_1。round2_045 不动(Batch 6 未修尾部,记遗留)。
"""
import sys, json, hashlib, collections, os
sys.path.insert(0, '.')

OFF = 'data/item_bank/v4/chemistry_v4_3329.jsonl'
CANDDIR = 'data/batch6_candidate_20260703'
STAGE = '/tmp/yher_batch7_stage'
NEW_SCHEMA = 'ws3_schema_v4_1'
os.makedirs(STAGE, exist_ok=True)

official = [json.loads(l) for l in open(OFF)]
assert len(official) == 3329
cand = {json.loads(l)['item_id']: json.loads(l) for l in open(f'{CANDDIR}/ws1_segmentation/fixed_candidate_items.jsonl')}
sign_inh = [json.loads(l) for l in open(f'{CANDDIR}/no_answer/claude_signoff_answer_inherit_20260704.jsonl')]
sign_exc = [json.loads(l) for l in open(f'{CANDDIR}/no_answer/claude_signoff_answerless_exclude_20260704.jsonl')]
omml_after = [json.loads(l) for l in open(f'{CANDDIR}/table_formula/omml_literal_after.jsonl')]

v3 = {}
for line in open('data/item_bank/chemistry_v3_6695.jsonl'):
    o = json.loads(line)
    for k in ('question_id', 'item_id'):
        if o.get(k): v3[o[k]] = o

SIGNER = 'claude_batch7_apply_20260704'
AUTH = '用户授权原话(2026-07-04):「授权batch7」,指 BATCH7_REAPPLY_PLAN_2026-07-04.md'

# ---- 集合构建 ----
naq = {it['item_id'] for it in official if it.get('group_key') == '答案202501'}
assert len(naq) == 19
naq |= {r['item_id'] for r in sign_exc if r['subtype'] == 'no_v3_candidate_not_a_question'}
assert len(naq) == 23

seg9 = {'5f4a44d4ade17fc93ed3ae64463dee085ec051e1','879f4b1994ed9752c47973b95788ac01f5cc0fe8',
        'c6cc5e8b2ca491c4a97ee273b334d19f545718ab','52ad9475e8ec03f35d310d0c45241b582937b403',
        '3c52fe3dde5c702ea5b4a0a237d35164b75f1cb1','016d8694492726cbe7c30e55a5fb7d5996aaab83',
        'cef99888ee63520a65f733dffaa7364aad1267b2','124a6deff2304e25a842682760c3fde18eeb371a',
        '50e085c5bbcfa9aefb59598796d4130e83836dac'}
seg9 -= naq
assert len(seg9) == 9, f'seg9={len(seg9)}'

ans_c = {r['item_id'] for r in sign_exc} - naq - seg9
assert len(ans_c) == 36, f'ans_c={len(ans_c)}'

inh7 = {r['item_id']: r for r in sign_inh}
assert len(inh7) == 7
swap = {}
for a in omml_after:
    if a.get('scope') == 'service' and a.get('rerun_status') == 'rerun_matched':
        assert a['new_item_id'] in cand, a
        swap[a['old_item_id']] = cand[a['new_item_id']]
assert len(swap) == 22, f'swap={len(swap)}'
R042 = '18517e489fdd049936f2212b928923db8b926737'

groups = [naq, seg9, ans_c, set(inh7), set(swap), {R042}]
for i in range(len(groups)):
    for j in range(i + 1, len(groups)):
        assert not (groups[i] & groups[j]), f'处置类交集非空: {i}/{j} {groups[i] & groups[j]}'

# ---- 逐行生成 ----
acct = collections.Counter()
out = []
for it in official:
    row = json.loads(json.dumps(it, ensure_ascii=False))  # 深拷贝
    iid = row['item_id']
    row['schema_version'] = NEW_SCHEMA
    if iid in naq:
        row['pool'] = 'excluded_not_a_question'
        row['service_eligible'] = False
        fl = row.setdefault('quality_flags', [])
        tag = 'not_a_question_answer_only_doc' if row.get('group_key') == '答案202501' else 'not_a_question_exam_instructions'
        if tag not in fl: fl.append(tag)
        row['pool_change'] = {'from': it['pool'], 'reason': tag, 'reviewer': SIGNER, 'authorization': AUTH}
        acct['A_not_a_question'] += 1
    elif iid in seg9:
        row['pool'] = 'excluded_bad_segmentation'
        row['service_eligible'] = False
        row['pool_change'] = {'from': it['pool'], 'reason': 'service_exclusions_2026-07-03_formalized',
                              'reviewer': SIGNER, 'authorization': AUTH}
        acct['B_bad_segmentation'] += 1
    elif iid in ans_c:
        row['pool'] = 'excluded_answerless'
        row['service_eligible'] = False
        row['pool_change'] = {'from': it['pool'], 'reason': 'claude_signoff_answerless_exclude_20260704',
                              'reviewer': SIGNER, 'authorization': AUTH}
        acct['C_answerless'] += 1
    elif iid in inh7:
        s = inh7[iid]
        vrow = v3[s['v3_item_id']]
        for f in ('kg_nodes', 'knowledge_points', 'rubric', 'standard_solution'):
            if vrow.get(f) is not None:
                row[f] = json.loads(json.dumps(vrow[f], ensure_ascii=False))
        sol = row.get('standard_solution') or {}
        assert [a for a in (sol.get('final_answers') or []) if str(a).strip()] or str(sol.get('standard_answer') or '').strip(), iid
        row['answer_available'] = True
        al = row.setdefault('alignment', {})
        al['status'] = 'manual_inherited'
        al['aligned_item_id'] = s['v3_item_id']
        al['manual_override'] = {'decision': 'manual_inherit_answer', 'evidence': s['evidence'],
                                 'reviewer': s['reviewer'], 'similarity_prescreen': s['similarity_prescreen'],
                                 'authorization': AUTH}
        acct['D_inherit'] += 1
    elif iid in swap:
        c = swap[iid]
        for f in ('stem_blocks', 'analysis_blocks', 'answer_blocks_effective', 'stem_text'):
            if f in c: row[f] = c[f]
        row['block_repair'] = {'batch': 'batch6_omml_table_fix', 'candidate_item_id': c['item_id'],
                               'note': 'blocks swapped in-place; id/alignment/pool preserved; stem_hash/stem_normalized kept (extraction-time fingerprints)',
                               'reviewer': SIGNER}
        acct['E_block_swap'] += 1
    elif iid == R042:
        al = row['alignment']
        al['status'] = 'manual_no_inherit'
        mr = al.setdefault('manual_resolution', {})
        mr['decision'] = 'manual_no_split_keep_composite'
        mr['reviewer'] = SIGNER
        mr['reason'] = '题已在服务池且有整题答案;manual_split_required 实为对齐问题;拆题成本>收益。KG 人工标注。'
        mr['authorization'] = AUTH
        row['kg_nodes'] = ['氧化还原反应', '杂化轨道', '化学键']
        acct['F_round2_042'] += 1
    else:
        acct['untouched_schema_bump_only'] += 1
    out.append(row)

assert len(out) == 3329
assert acct['A_not_a_question'] == 23 and acct['B_bad_segmentation'] == 9
assert acct['C_answerless'] == 36 and acct['D_inherit'] == 7
assert acct['E_block_swap'] == 22 and acct['F_round2_042'] == 1

stage_path = f'{STAGE}/chemistry_v4_1_3329.jsonl'
with open(stage_path, 'w') as f:
    for row in out:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print('=== 账目 ===')
for k, v in sorted(acct.items()):
    print(f'  {k}: {v}')
print('  合计:', sum(acct.values()), '== 3329')
print('staged:', stage_path)

# ---- 未动行逐字节门 ----
touched = naq | seg9 | ans_c | set(inh7) | set(swap) | {R042}
byid_new = {r['item_id']: r for r in out}
diff = 0
for it in official:
    if it['item_id'] in touched: continue
    a = json.loads(json.dumps(it, ensure_ascii=False)); a['schema_version'] = NEW_SCHEMA
    if json.dumps(a, sort_keys=True, ensure_ascii=False) != json.dumps(byid_new[it['item_id']], sort_keys=True, ensure_ascii=False):
        diff += 1
print('未动行(除 schema 外)逐字节 diff:', diff, '(必须 0)')
assert diff == 0
print('STAGE_BUILD_PASS')
