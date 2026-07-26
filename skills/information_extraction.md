# InformationExtractionSkill

```json
{
  "skill_name": "InformationExtractionSkill",
  "query_types": ["information_extraction"],
  "required_slots": [],
  "input_schema": {"question": "str", "collection_name": "str"},
  "tool_chain": ["clarify_gate", "query_expander", "parallel_hybrid_retrieval", "two_stage_hybrid_rerank", "evidence_gate", "answer_generator"],
  "output_schema": {"answer": "str", "citations": "list", "evidence": "list", "decision": "str"},
  "guardrails": {"refuse_on_low_evidence": true},
  "trace_fields": ["selected_skill", "tool_chain", "observations", "gate_decision"],
  "few_shot_examples": [{"question": "芯导科技 2025 年营业收入是多少？", "answer_style": "direct evidence-grounded answer"}],
  "slot_schema": {"required": [], "optional": ["company", "years", "metric", "scope"]},
  "tool_constraints": {"allowed_tools": ["query_expander", "retriever", "reranker", "answer_generator"]},
  "execution_config": {"max_iterations": 1, "retry_on_low_evidence": true}
}
```

## Task Description
Extract an explicitly disclosed fact, text, or value from annual-report evidence.

## Prompt Template
Return the requested disclosed information directly. Preserve company, period, metric, value, unit, and citation alignment. Do not add analysis that the user did not request.
