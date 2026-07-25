# ComparisonSkill

```json
{
  "skill_name": "ComparisonSkill",
  "query_types": ["comparison"],
  "required_slots": ["compare_targets"],
  "input_schema": {"question": "str", "compare_targets": "list", "collection_name": "str"},
  "tool_chain": ["clarify_gate", "query_expander", "parallel_hybrid_retrieval", "two_stage_hybrid_rerank", "evidence_gate", "answer_generator"],
  "output_schema": {"answer": "str", "citations": "list", "evidence": "list", "decision": "str"},
  "guardrails": {"require_multi_scope_evidence": true},
  "trace_fields": ["selected_skill", "tool_chain", "observations", "scope_coverage"],
  "few_shot_examples": [{"question": "比较芯导科技 2024 年和 2025 年的营业收入。", "expected_slots": {"compare_targets": ["2024", "2025"]}}],
  "slot_schema": {"required": ["compare_targets"], "optional": ["period", "years", "metric", "scope"]},
  "tool_constraints": {"min_compare_targets": 2},
  "execution_config": {"comparison_layout": "side_by_side"}
}
```

## Task Description
Compare at least two companies, periods, documents, or business objects using aligned annual-report evidence.

## Prompt Template
Keep the comparison scope and metric definitions aligned. Present each target's evidence before stating similarities, differences, or ranking.
