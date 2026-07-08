from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

from utils.config_loader import PROJECT_ROOT, load_yaml_file


DEFAULT_COMPANY_REGISTRY_PATH = PROJECT_ROOT / "config" / "company_registry.yaml"


@dataclass(frozen=True)
class CompanyProfile:
    company_id: str
    company_name: str
    aliases: tuple[str, ...] = ()

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "company_id": self.company_id,
            "company_name": self.company_name,
            "company_aliases": list(self.aliases),
        }


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _normalize_company_id(value: Any) -> str:
    text = _clean(value).lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


class CompanyRegistry:
    def __init__(self, companies: Iterable[Mapping[str, Any]] | None = None) -> None:
        self._profiles: Dict[str, CompanyProfile] = {}
        for item in companies or self._configured_companies():
            if not isinstance(item, Mapping):
                continue
            profile = self._profile_from_mapping(item)
            if profile is not None:
                self._profiles[profile.company_id] = profile

    @staticmethod
    def _configured_companies() -> List[Mapping[str, Any]]:
        config = load_yaml_file(DEFAULT_COMPANY_REGISTRY_PATH)
        raw = config.get("companies", []) if isinstance(config, Mapping) else []
        return list(raw or []) if isinstance(raw, list) else []

    @staticmethod
    def _profile_from_mapping(item: Mapping[str, Any]) -> CompanyProfile | None:
        company_name = _clean(item.get("company_name") or item.get("name") or item.get("company"))
        company_id = _clean(item.get("company_id") or item.get("id")) or _normalize_company_id(company_name)
        if not company_id or not company_name:
            return None
        aliases = []
        for value in list(item.get("aliases") or item.get("company_aliases") or []):
            alias = _clean(value)
            if alias and alias not in aliases and alias != company_name:
                aliases.append(alias)
        return CompanyProfile(company_id=company_id, company_name=company_name, aliases=tuple(aliases))

    def resolve(self, company_id: str = "", company_name: str = "") -> CompanyProfile | None:
        key = _clean(company_id)
        if key and key in self._profiles:
            return self._profiles[key]
        name = _clean(company_name)
        if not name:
            return None
        for profile in self._profiles.values():
            if name == profile.company_name or name in profile.aliases:
                return profile
        generated_id = _normalize_company_id(name)
        if generated_id:
            return CompanyProfile(company_id=generated_id, company_name=name)
        return None

    def list_profiles(self) -> List[CompanyProfile]:
        return sorted(self._profiles.values(), key=lambda profile: profile.company_id)

    def match_question(self, question: str, document_scopes: Iterable[Mapping[str, Any]] | None = None) -> CompanyProfile | None:
        text = _clean(question)
        if not text:
            return None
        candidates: Dict[str, CompanyProfile] = dict(self._profiles)
        for item in document_scopes or []:
            profile = self._profile_from_mapping(item)
            if profile is not None and profile.company_id not in candidates:
                candidates[profile.company_id] = profile
        matches: list[tuple[int, CompanyProfile]] = []
        for profile in candidates.values():
            names = [profile.company_name, *profile.aliases, profile.company_id]
            best = max((len(name) for name in names if name and name in text), default=0)
            if best:
                matches.append((best, profile))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0], reverse=True)
        return matches[0][1]


_DEFAULT_REGISTRY: CompanyRegistry | None = None


def get_company_registry() -> CompanyRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = CompanyRegistry()
    return _DEFAULT_REGISTRY
