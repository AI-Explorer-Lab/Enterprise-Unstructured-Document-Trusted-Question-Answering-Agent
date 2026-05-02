from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from service.agent.skills import load_all_skills, load_skill_definition


class MarkdownSkillLoaderTestCase(unittest.TestCase):
    def test_markdown_skill_loads_required_metadata(self):
        skill = load_skill_definition(Path("skills/table_qa.md"))

        package = skill.package_metadata()
        self.assertEqual(package["skill_name"], "TableQASkill")
        self.assertEqual(package["query_types"], ["table_qa"])
        self.assertIn("metric", skill.required_slots)
        self.assertTrue(package["tool_chain"])
        self.assertTrue(package["guardrails"])
        self.assertTrue(package["slot_schema"])
        self.assertTrue(package["execution_config"])
        self.assertTrue(package["task_description"])
        self.assertTrue(package["prompt_template"])
        self.assertTrue(package["few_shot_examples"])
        self.assertTrue(package["tool_constraints"])

    def test_markdown_skill_missing_required_field_raises(self):
        with tempfile.TemporaryDirectory() as td:
            broken = Path(td) / "broken.md"
            broken.write_text(
                """
# BrokenSkill

```json
{
  "skill_name": "BrokenSkill",
  "query_types": ["fact_lookup"],
  "required_slots": [],
  "tool_chain": ["RetrievalEngine.search"],
  "guardrails": {},
  "execution_config": {},
  "few_shot_examples": [],
  "tool_constraints": {}
}
```

## Task Description
desc

## Prompt Template
prompt
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing required fields"):
                load_skill_definition(broken)

    def test_markdown_skills_duplicate_name_raises(self):
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            content = """
# Demo

```json
{
  "skill_name": "SameName",
  "query_types": ["fact_lookup"],
  "required_slots": [],
  "tool_chain": ["RetrievalEngine.search"],
  "guardrails": {},
  "slot_schema": {},
  "execution_config": {},
  "few_shot_examples": [],
  "tool_constraints": {},
  "task_description": "desc",
  "prompt_template": "prompt"
}
```
""".strip()
            (temp_dir / "a.md").write_text(content, encoding="utf-8")
            (temp_dir / "b.md").write_text(content, encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Duplicate skill_name"):
                load_all_skills(temp_dir)


if __name__ == "__main__":
    unittest.main()
