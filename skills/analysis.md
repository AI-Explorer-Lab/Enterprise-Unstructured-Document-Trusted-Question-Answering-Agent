# AnalysisSkill

```json
{
  "skill_name": "AnalysisSkill",
  "query_types": ["analysis"],
  "required_slots": ["scope"],
  "input_schema": {"question": "str", "scope": "str", "collection_name": "str"},
  "tool_chain": ["clarify_gate", "query_expander", "parallel_hybrid_retrieval", "two_stage_hybrid_rerank", "evidence_gate", "answer_generator"],
  "output_schema": {"answer": "str", "citations": "list", "evidence": "list", "decision": "str"},
  "guardrails": {"separate_evidence_from_inference": true},
  "trace_fields": ["selected_skill", "tool_chain", "observations", "analysis_scope"],
  "few_shot_examples": [{"question": "分析经营现金流下降的原因和影响。", "expected_slots": {"scope": "经营现金流下降的原因和影响"}}],
  "slot_schema": {"required": ["scope"], "optional": ["period", "years", "metric", "focus"]},
  "tool_constraints": {"require_multiple_supporting_evidence": true},
  "execution_config": {"analysis_style": "evidence_then_inference"}
}
```

## Task Description
Explain causes, impacts, trends, risks, or implications using annual-report evidence.

## Prompt Template
Separate disclosed facts from analytical inference. Explain the evidence chain and do not present unsupported causality or judgment as fact.
