# MetricCalculationSkill

```json
{
  "skill_name": "MetricCalculationSkill",
  "query_types": ["metric_calculation"],
  "required_slots": ["metric"],
  "input_schema": {"question": "str", "metric": "str", "period": "str", "collection_name": "str"},
  "tool_chain": ["clarify_gate", "query_expander", "parallel_hybrid_retrieval", "table_prioritized_retrieval", "two_stage_hybrid_rerank", "evidence_gate", "answer_generator"],
  "output_schema": {"answer": "str", "formula": "str", "inputs": "list", "citations": "list", "evidence": "list", "decision": "str"},
  "guardrails": {"require_numeric_inputs": true, "show_formula": true},
  "trace_fields": ["selected_skill", "tool_chain", "observations", "calculation_inputs"],
  "few_shot_examples": [{"question": "根据 2024 年和 2025 年营业收入计算同比增长率。", "expected_slots": {"metric": "营业收入增长率", "period": "2024、2025"}}],
  "slot_schema": {"required": ["metric"], "optional": ["period", "years", "unit", "scope"]},
  "tool_constraints": {"must_use_evidence_values": true, "must_show_formula": true},
  "execution_config": {"table_priority": "high", "retry_on_missing_inputs": true}
}
```

## Task Description
Calculate a derived financial metric from values grounded in annual-report evidence.

## Prompt Template
Identify the evidence-backed numeric inputs, state the formula, calculate the result, preserve units and periods, and cite every input. Refuse when required inputs are missing.
