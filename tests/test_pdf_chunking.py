from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from service.embedding.embedding_service import EMBEDDING_DIMENSION, EmbeddingService
from service.pdf.heading_recovery import detect_heading_level
from service.pdf.mineru_client import MinerUClient
from service.pdf.mineru_parser import parse_mineru_payload
from service.pdf.pdf_loader import PdfLoaderError, collect_pdf_paths
from service.pdf.structured_chunker import ChunkingConfig, StructuredChunker

try:  # pragma: no cover - optional dependency for richer local fallback tests
    import fitz
except Exception:  # pragma: no cover
    fitz = None


@pytest.fixture
def workspace_tmp_dir() -> Path:
    root = Path.cwd() / ".tmp_pytest_workspace"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = Path(tempfile.mkdtemp(prefix="pdf_chunk_", dir=str(root)))
    try:
        yield case_dir
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def _create_pdf(path: Path, lines: list[str]) -> None:
    if fitz is None:
        minimal_payload = "%PDF-1.4\n" + "\n".join(lines) + "\n%%EOF\n"
        path.write_bytes(minimal_payload.encode("utf-8"))
        return

    document = fitz.open()
    page = document.new_page()
    y = 72
    for line in lines:
        page.insert_text((72, y), line)
        y += 20
    document.save(str(path))
    document.close()


def _build_text_block(text: str, index: int, block_type: str = "text") -> dict:
    return {
        "index": index,
        "type": block_type,
        "lines": [{"spans": [{"content": text}]}],
    }


def test_pdf_loader_accepts_pdf_file_and_directory(workspace_tmp_dir: Path) -> None:
    pdf_a = workspace_tmp_dir / "a.pdf"
    pdf_b = workspace_tmp_dir / "b.pdf"
    txt_file = workspace_tmp_dir / "notes.txt"

    _create_pdf(pdf_a, ["第1节 总则", "一、目的", "（一）范围"])
    _create_pdf(pdf_b, ["第2节 范围", "1、普通枚举"])
    txt_file.write_text("not a pdf", encoding="utf-8")

    single = collect_pdf_paths(pdf_a)
    assert single == [pdf_a.resolve()]

    directory = collect_pdf_paths(workspace_tmp_dir)
    assert directory == sorted([pdf_a.resolve(), pdf_b.resolve()], key=lambda item: str(item).lower())

    with pytest.raises(PdfLoaderError):
        collect_pdf_paths(txt_file)


def test_mineru_client_supports_json_path_fallback_and_cache(workspace_tmp_dir: Path) -> None:
    pdf_path = workspace_tmp_dir / "report.pdf"
    _create_pdf(pdf_path, ["第1节 经营情况", "一、核心指标", "（一）收入增长"])

    client = MinerUClient(cache_ttl_seconds=120, cache_max_items=8, remote_enabled=False)

    json_payload = {
        "source": "mineru_json_file",
        "pdf_info": [
            {
                "page_idx": 0,
                "para_blocks": [_build_text_block("第1节 从JSON读取", index=0, block_type="title")],
            }
        ],
    }
    json_path = workspace_tmp_dir / "report.json"
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False), encoding="utf-8")

    payload_from_json = client.parse_pdf_to_mineru_json(pdf_path, mineru_json_path=json_path, use_cache=False)
    assert payload_from_json["source"] == "mineru_json_file"

    fallback_payload_first = client.parse_pdf_to_mineru_json(pdf_path, use_cache=True, force_rebuild=True)
    fallback_payload_second = client.parse_pdf_to_mineru_json(pdf_path, use_cache=True, force_rebuild=False)
    assert "pdf_info" in fallback_payload_first
    assert fallback_payload_first == fallback_payload_second
    assert len(client.document_parse_cache) == 1


def test_mineru_parser_restores_reading_order_and_extracts_types() -> None:
    payload = {
        "pdf_info": [
            {
                "page_idx": 1,
                "para_blocks": [_build_text_block("后页正文", index=3, block_type="text")],
            },
            {
                "page_idx": 0,
                "para_blocks": [
                    _build_text_block("一、先出现标题", index=1, block_type="title"),
                    _build_text_block("列表项", index=2, block_type="list"),
                    {
                        "index": 4,
                        "type": "table",
                        "table_rows": [["指标", "数值"], ["收入", "100"]],
                    },
                ],
            },
        ]
    }

    blocks = parse_mineru_payload(payload)
    assert [(item["page_idx"], item["block_index"]) for item in blocks] == [(0, 1), (0, 2), (0, 4), (1, 3)]
    assert [item["type"] for item in blocks] == ["title", "list", "table", "text"]
    assert blocks[2]["rows"][0] == ["指标", "数值"]


def test_heading_recovery_rule_mapping() -> None:
    assert detect_heading_level("第1节 总则") == 1
    assert detect_heading_level("一、适用范围") == 2
    assert detect_heading_level("（一）定义") == 3
    assert detect_heading_level("1、普通枚举") == 0


def test_structured_chunker_splits_table_and_preserves_metadata() -> None:
    parsed_blocks = [
        {"type": "title", "page_idx": 0, "block_index": 0, "text": "第1节 财务总览"},
        {
            "type": "text",
            "page_idx": 0,
            "block_index": 1,
            "text": "一、收入说明 2025年收入增长明显，主要来自企业客户与海外市场。",
        },
        {
            "type": "table",
            "page_idx": 0,
            "block_index": 2,
            "table_id": "table_finance",
            "rows": [
                ["指标", "Q1", "Q2"],
                ["收入", "100", "120"],
                ["毛利率", "25%", "28%"],
                ["研发费用", "30", "35"],
                ["销售费用", "20", "24"],
                ["管理费用", "15", "16"],
            ],
        },
    ]

    chunker = StructuredChunker(
        ChunkingConfig(
            chunk_size_tokens=24,
            chunk_overlap_tokens=4,
            max_chunk_size_tokens=36,
        )
    )
    chunks = chunker.chunk_parsed_blocks(
        parsed_blocks=parsed_blocks,
        doc_id="doc_finance",
        collection_name="finance_2025",
        doc_source="finance_report.pdf",
    )

    assert any(chunk["chunk_type"] == "text" for chunk in chunks)
    table_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "table"]
    assert len(table_chunks) >= 2

    expected_header = "指标 | Q1 | Q2"
    for chunk in table_chunks:
        assert chunk["table_header_text"] == expected_header
        assert chunk["embedding_dim"] == EMBEDDING_DIMENSION
        assert chunk["sub_table_id"].startswith("table_finance_")
        assert chunk["heading_path"].startswith("第1节")
        assert chunk["collection_name"] == "finance_2025"


def test_embedding_service_async_embedding_and_cache() -> None:
    service = EmbeddingService(cache_ttl_seconds=120, cache_max_items=16)

    first_vector = asyncio.run(service.embed_text("alpha beta gamma"))
    second_vector = asyncio.run(service.embed_text("alpha beta gamma"))
    assert len(first_vector) == EMBEDDING_DIMENSION
    assert first_vector == second_vector
    assert len(service.embedding_cache) == 1

    chunk_vector = asyncio.run(service.embed_text("table row content", chunk_text=True))
    assert len(chunk_vector) == EMBEDDING_DIMENSION
    assert len(service.chunk_embedding_cache) == 1

    vectors = asyncio.run(service.embed_texts(["alpha", "beta", "alpha"], max_concurrency=2))
    assert len(vectors) == 3
    assert vectors[0] == vectors[2]
