from __future__ import annotations

import unittest

from service.agent.skill_registry import DEFAULT_SKILL_REGISTRY
from service.agent.skills import load_all_skills


class SkillRegistryTestCase(unittest.TestCase):
    def test_registry_exposes_skill_package_metadata(self):
        skill = DEFAULT_SKILL_REGISTRY.select_skill("table_qa")
        package = DEFAULT_SKILL_REGISTRY.get_skill_package(skill.skill_name)

        self.assertEqual(skill.skill_name, "TableQASkill")
        self.assertIsInstance(package, dict)
        self.assertEqual(package["skill_name"], "TableQASkill")
        self.assertIn("slot_schema", package)
        self.assertIn("tool_constraints", package)
        self.assertIn("execution_config", package)

    def test_markdown_skill_loader_returns_expected_skill_count(self):
        skills = load_all_skills()
        names = {skill.skill_name for skill in skills}

        self.assertEqual(len(skills), 6)
        self.assertIn("TableQASkill", names)
        self.assertIn("prompt_template", skills[0].package_metadata())


if __name__ == "__main__":
    unittest.main()
