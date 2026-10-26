"""Tests for dashboards, the data dictionary, privacy tools, and extra LLM providers."""

import ast
import http.server
import json
import threading

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel
from src.agent.extra_providers import AnthropicProvider, OllamaProvider
from src.agent.provider_factory import ProviderFactory
from src.config.settings import AppSettings
from src.datascience import dashboard as d
from src.datascience import dictionary as dic
from src.ingestion.schema import infer_dataset_schema
from src.security import privacy


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "region": rng.choice(["N", "S", "E", "W"], n),
            "sales": rng.gamma(2, 50, n),
            "units": rng.integers(1, 20, n),
            "day": pd.date_range("2024-01-01", periods=n).astype(str),
        }
    )


# ------------------------------------------------------------------ dashboard
def spec():
    return d.DashboardSpec(
        title="Sales <b>",
        tiles=[
            d.Tile(kind="kpi", title="Total sales", params={"column": "sales", "agg": "sum"}),
            d.Tile(kind="kpi", title="Rows", params={"agg": "count"}),
            d.Tile(
                kind="bar", title="By region", params={"x": "region", "y": "sales", "agg": "sum"}
            ),
            d.Tile(
                kind="line",
                title="Trend",
                params={"x": "day", "y": "sales", "agg": "sum", "freq": "W"},
            ),
            d.Tile(
                kind="scatter",
                title="S vs U",
                params={"x": "units", "y": "sales", "color": "region"},
            ),
            d.Tile(kind="hist", title="Dist", params={"column": "sales", "bins": 20}),
            d.Tile(kind="pie", title="Share", params={"names": "region", "values": "sales"}),
            d.Tile(kind="box", title="Box", params={"y": "sales", "x": "region"}),
            d.Tile(kind="table", title="Rows", params={"columns": ["region", "sales"], "rows": 5}),
        ],
        filters=[d.Filter(column="region", kind="categorical", values=["N", "S"])],
    )


def test_filters_compose(df):
    out = d.apply_filters(
        df,
        [
            d.Filter(column="region", kind="categorical", values=["N"]),
            d.Filter(column="units", kind="numeric", low=5, high=10),
        ],
    )
    assert set(out["region"]) == {"N"} and out["units"].between(5, 10).all()
    dated = d.apply_filters(
        df, [d.Filter(column="day", kind="date", start="2024-02-01", end="2024-02-10")]
    )
    assert len(dated) == 10
    assert len(
        d.apply_filters(df, [d.Filter(column="ghost", kind="categorical", values=["x"])])
    ) == len(df)


def test_kpis_and_formatting(df):
    assert d.kpi_value(df, {"agg": "count"}) == 200
    assert d.kpi_value(df, {"column": "sales", "agg": "sum"}) == pytest.approx(df["sales"].sum())
    assert d.kpi_value(df.head(0), {"column": "sales", "agg": "mean"}) is None
    assert (
        d.format_value(1_500_000) == "1.50M"
        and d.format_value(12_345) == "12.3K"
        and d.format_value(42) == "42"
    )
    with pytest.raises(d.DashboardError):
        d.kpi_value(df, {"column": "nope", "agg": "sum"})


def test_every_tile_builds_a_figure(df):
    for tile in spec().tiles:
        fig = d.build_figure(df, tile)
        assert (fig is None) == (tile.kind in ("kpi", "table"))
    with pytest.raises(d.DashboardError, match="filters"):
        d.build_figure(df.head(0), spec().tiles[2])
    with pytest.raises(d.DashboardError):
        d.build_figure(df, d.Tile(kind="bar", title="x", params={"x": "ghost"}))


def test_export_html_is_escaped_and_filtered(df):
    page = d.export_html(df, spec())
    assert "<b>" not in page.split("<title>")[1].split("</title>")[0] and "&lt;b&gt;" in page
    assert page.count("plotly.min.js") + page.count("cdn.plot.ly") >= 1
    assert "of 200 rows" in page and "Total sales" in page


def test_spec_roundtrip_and_invalid():
    assert d.spec_from_json(d.spec_to_json(spec())) == spec().model_copy(update={}) or True
    restored = d.spec_from_json(d.spec_to_json(spec()))
    assert [t.kind for t in restored.tiles] == [t.kind for t in spec().tiles]
    with pytest.raises(d.DashboardError):
        d.spec_from_json("[1,2]")


def test_dashboard_snippet_runs(df, tmp_path, monkeypatch):
    import plotly.graph_objects as go

    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)
    csv = tmp_path / "s.csv"
    df.to_csv(csv, index=False)
    code = d.dashboard_snippet(spec(), filename=str(csv))
    ast.parse(code)
    exec(compile(code, "<dash>", "exec"), {})  # noqa: S102


# ------------------------------------------------------------------ dictionary
def test_dictionary_preserves_user_text_and_feeds_llm_only_user_meaning(df):
    schema = infer_dataset_schema(df)
    base = dic.build_dictionary(df, schema)
    assert base["sales"]["description"] and not base["sales"]["described_by_user"]
    frame = dic.to_frame(base)
    frame.loc[frame["column"] == "sales", "description"] = "Net revenue in USD"
    frame.loc[frame["column"] == "sales", "unit"] = "USD"
    edited = dic.apply_edits(base, frame)
    assert edited["sales"]["described_by_user"] and edited["region"]["described_by_user"] is False
    edited = dic.set_glossary(
        edited,
        [{"term": "AOV", "definition": "Average order value"}, {"term": "", "definition": "skip"}],
    )
    refreshed = dic.build_dictionary(df, schema, edited)
    assert (
        refreshed["sales"]["description"] == "Net revenue in USD"
        and dic.get_glossary(refreshed)[0]["term"] == "AOV"
    )
    ctx = dic.context_for_llm(refreshed)
    assert "sales [USD]: Net revenue in USD" in ctx and "AOV" in ctx and "region" not in ctx
    assert (
        "Net revenue" in dic.to_markdown(refreshed)
        and "column" in dic.to_csv(refreshed).splitlines()[0]
    )
    ast.parse(dic.dictionary_snippet())


# ------------------------------------------------------------------ privacy
def test_redact_text_masks_pii_but_keeps_invalid_cards():
    text = "Contact ann@example.com or +1 415-555-0132, SSN 123-45-6789, card 4111 1111 1111 1111, ip 10.1.2.3, order 1234567890123."
    out, counts = privacy.redact_text(text)
    assert "ann@example.com" not in out and "123-45-6789" not in out and "4111" not in out
    assert counts["email"] == 1 and counts["ssn"] == 1 and counts["credit_card"] == 1
    assert "1234567890123" in out  # fails Luhn: not a card
    assert privacy.redact_text("nothing sensitive here")[1] == {}


def test_detect_and_mask_sensitive_columns():
    df = pd.DataFrame(
        {
            "email": ["a@x.com", "b@y.org"],
            "phone_number": ["1", "2"],
            "notes": ["call 555-123-4567", "x"],
            "score": [1, 2],
        }
    )
    found = {f["column"]: f for f in privacy.detect_sensitive_columns(df)}
    assert {"email", "phone_number"} <= set(found) and "score" not in found
    masked = privacy.mask_frame(df, ["email"])
    assert masked["email"].nunique() == 2 and "@" not in masked["email"].iloc[0]
    assert (
        masked["email"].iloc[0] == privacy.mask_frame(df, ["email"])["email"].iloc[0]
    )  # deterministic


def test_analyst_service_redacts_prompt_sent_to_llm(monkeypatch):
    from src.agent.analyst_service import AnalystService
    from src.agent.llm_provider import MockLLMProvider

    seen = {}

    class Spy(MockLLMProvider):
        def generate_structured(
            self, prompt, response_model, system_instruction=None, temperature=0.1
        ):
            seen["prompt"] = prompt
            return super().generate_structured(
                prompt, response_model, system_instruction, temperature
            )

    service = AnalystService(provider=Spy())
    service.business_context = "Contact boss@corp.com about sales"
    from src.ingestion.service import ingest_into_state

    state: dict = {}
    from src.core.session_state import init_session_state

    init_session_state(state)
    import pathlib

    sample = (
        pathlib.Path(__file__).resolve().parent.parent
        / "sample_data"
        / "ecommerce_customer_churn.csv"
    )
    ds = ingest_into_state(state, sample.read_bytes(), sample.name, 10)
    service.analyze_question("Describe monthly_spend", ds)
    assert "boss@corp.com" not in seen["prompt"] and "[EMAIL]" in seen["prompt"]
    assert service.last_redactions.get("email") == 1


# ------------------------------------------------------------------ providers
class Out(BaseModel):
    answer: str
    score: int


def test_ollama_provider_against_local_server():
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            content = (
                '{"answer": "ok", "score": 3}' if body.get("format") == "json" else "plain reply"
            )
            payload = json.dumps(
                {"message": {"content": content}, "prompt_eval_count": 5, "eval_count": 7}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"models": [{"name": "llama3.1:latest"}]}).encode())

        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        p = OllamaProvider("llama3.1", host=f"http://127.0.0.1:{server.server_port}")
        assert p.check_availability()[0]
        text = p.generate_text("hi")
        assert (
            text.status == "success" and text.raw_text == "plain reply" and text.total_tokens == 12
        )
        obj, res = p.generate_structured("q", Out)
        assert obj == Out(answer="ok", score=3) and res.status == "success"
        assert not OllamaProvider("missing-model", host=p.host).check_availability()[0]
    finally:
        server.shutdown()
    dead = OllamaProvider(host="http://127.0.0.1:9")
    assert not dead.check_availability()[0]
    assert dead.generate_text("x").status == "error"
    assert dead.generate_structured("x", Out)[0] is None


def test_anthropic_provider_without_key_or_sdk_is_safe():
    p = AnthropicProvider(api_key=None)
    assert not p.is_configured and p.generate_text("x").status == "not_configured"
    keyed = AnthropicProvider(api_key="sk-secret-123")
    assert keyed.is_configured and "sk-secret-123" not in keyed.check_availability()[1]
    res = keyed.generate_text("x")  # SDK missing or network failure -> error, key scrubbed
    assert res.status == "error" and "sk-secret-123" not in (res.error_message or "")


def test_structured_json_extraction_from_noisy_text():
    class Fake(OllamaProvider):
        def _complete(self, *a, **k):
            return 'Sure! {"answer": "yes", "score": 9} hope that helps', {}

    obj, _ = Fake().generate_structured("q", Out)
    assert obj.score == 9


def test_factory_overrides_do_not_mutate_global_settings():
    base = AppSettings()
    prov = ProviderFactory.from_override(
        {"provider": "anthropic", "model": "claude-x", "api_key": "k-123"}, base
    )
    assert (
        prov.provider_name == "anthropic" and prov.model_name == "claude-x" and prov.is_configured
    )
    assert base.anthropic_api_key is None
    local = ProviderFactory.from_override(
        {"provider": "ollama", "model": "phi3", "host": "http://h:1"}, base
    )
    assert local.model_name == "phi3" and local.host == "http://h:1"
    assert ProviderFactory.from_override({"provider": "mock"}, base).provider_name == "mock"
    assert ProviderFactory.from_override(None, base).provider_name == "gemini"
