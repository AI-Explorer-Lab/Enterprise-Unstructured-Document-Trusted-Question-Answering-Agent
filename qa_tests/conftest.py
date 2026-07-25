from __future__ import annotations

import sys
import types


database_module = types.ModuleType("database")
database_module.get_async_session = lambda *args, **kwargs: None
sys.modules.setdefault("database", database_module)

database_init_module = types.ModuleType("database.init_db")


async def _noop_init_pgvector_schema(*args, **kwargs):
    return None


database_init_module.init_pgvector_schema = _noop_init_pgvector_schema
sys.modules.setdefault("database.init_db", database_init_module)
