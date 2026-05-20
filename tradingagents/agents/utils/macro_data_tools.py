import calendar
import json
import math
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any

import requests
from langchain_core.tools import tool

from tradingagents.dataflows.config import get_config


FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
ECOS_API_ROOT = "https://ecos.bok.or.kr/api/StatisticSearch"

FRED_SERIES = {
    "US": [
        {
            "id": "FEDFUNDS",
            "name": "US Effective Federal Funds Rate",
            "frequency": "monthly",
            "unit": "percent",
        },
        {
            "id": "DGS2",
            "name": "US 2Y Treasury Yield",
            "frequency": "daily",
            "unit": "percent",
        },
        {
            "id": "DGS10",
            "name": "US 10Y Treasury Yield",
            "frequency": "daily",
            "unit": "percent",
        },
        {
            "id": "CPIAUCSL",
            "name": "US CPI All Urban Consumers",
            "frequency": "monthly",
            "unit": "index",
        },
        {
            "id": "UNRATE",
            "name": "US Unemployment Rate",
            "frequency": "monthly",
            "unit": "percent",
        },
        {
            "id": "DCOILWTICO",
            "name": "WTI Crude Oil Spot Price",
            "frequency": "daily",
            "unit": "usd_per_barrel",
        },
        {
            "id": "VIXCLS",
            "name": "CBOE VIX",
            "frequency": "daily",
            "unit": "index",
        },
        {
            "id": "DEXKOUS",
            "name": "USD/KRW Exchange Rate",
            "frequency": "daily",
            "unit": "krw_per_usd",
        },
    ],
    "GLOBAL": [
        {
            "id": "DGS10",
            "name": "US 10Y Treasury Yield",
            "frequency": "daily",
            "unit": "percent",
        },
        {
            "id": "DCOILWTICO",
            "name": "WTI Crude Oil Spot Price",
            "frequency": "daily",
            "unit": "usd_per_barrel",
        },
        {
            "id": "VIXCLS",
            "name": "CBOE VIX",
            "frequency": "daily",
            "unit": "index",
        },
        {
            "id": "DEXKOUS",
            "name": "USD/KRW Exchange Rate",
            "frequency": "daily",
            "unit": "krw_per_usd",
        },
    ],
}


# ECOS series and item codes were verified through the official ECOS
# StatisticTableList and StatisticItemList API metadata on 2026-05-21.
ECOS_SERIES = {
    "KR": [
        {
            "id": "722Y001:0101000",
            "stat_code": "722Y001",
            "item_code": "0101000",
            "name": "한국은행 기준금리",
            "frequency": "monthly",
            "cycle": "M",
            "unit": "percent",
        },
        {
            "id": "731Y001:0000001",
            "stat_code": "731Y001",
            "item_code": "0000001",
            "name": "USD/KRW 원/미국달러 매매기준율",
            "frequency": "daily",
            "cycle": "D",
            "unit": "krw_per_usd",
        },
        {
            "id": "817Y002:010190000",
            "stat_code": "817Y002",
            "item_code": "010190000",
            "name": "국고채 1년 금리",
            "frequency": "daily",
            "cycle": "D",
            "unit": "percent",
        },
        {
            "id": "817Y002:010200001",
            "stat_code": "817Y002",
            "item_code": "010200001",
            "name": "국고채 5년 금리",
            "frequency": "daily",
            "cycle": "D",
            "unit": "percent",
        },
        {
            "id": "817Y002:010210000",
            "stat_code": "817Y002",
            "item_code": "010210000",
            "name": "국고채 10년 금리",
            "frequency": "daily",
            "cycle": "D",
            "unit": "percent",
        },
        {
            "id": "901Y009:0",
            "stat_code": "901Y009",
            "item_code": "0",
            "name": "한국 소비자물가지수 총지수",
            "frequency": "monthly",
            "cycle": "M",
            "unit": "index",
        },
        {
            "id": "901Y118:T002",
            "stat_code": "901Y118",
            "item_code": "T002",
            "name": "수출금액",
            "frequency": "monthly",
            "cycle": "M",
            "unit": "usd_thousand",
        },
        {
            "id": "901Y118:T004",
            "stat_code": "901Y118",
            "item_code": "T004",
            "name": "수입금액",
            "frequency": "monthly",
            "cycle": "M",
            "unit": "usd_thousand",
        },
        {
            "id": "512Y013:99988",
            "stat_code": "512Y013",
            "item_code": "99988",
            "name": "기업경기실사지수 전산업 실적",
            "frequency": "monthly",
            "cycle": "M",
            "unit": "index",
        },
        {
            "id": "901Y067:I16A",
            "stat_code": "901Y067",
            "item_code": "I16A",
            "name": "경기선행종합지수",
            "frequency": "monthly",
            "cycle": "M",
            "unit": "index",
        },
        {
            "id": "901Y067:I16B",
            "stat_code": "901Y067",
            "item_code": "I16B",
            "name": "경기동행종합지수",
            "frequency": "monthly",
            "cycle": "M",
            "unit": "index",
        },
    ]
}


def _normalize_market(market: str) -> str:
    value = (market or "").strip().upper()
    if value in {"US", "USA", "UNITED STATES"}:
        return "US"
    if value in {"KR", "KOREA", "KOR", "SOUTH KOREA"}:
        return "KR"
    return "GLOBAL"


def _parse_as_of_date(curr_date: str) -> date:
    return datetime.strptime(curr_date, "%Y-%m-%d").date()


def _today_utc() -> date:
    return datetime.utcnow().date()


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _subtract_months(value: date, months: int) -> date:
    month_index = value.month - 1 - months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _period_start(as_of_date: date, lookback_months: int, cycle: str) -> str:
    start = _subtract_months(as_of_date, lookback_months + 1)
    if cycle == "D":
        return start.strftime("%Y%m%d")
    return start.strftime("%Y%m")


def _period_end(as_of_date: date, cycle: str) -> str:
    if cycle == "D":
        return as_of_date.strftime("%Y%m%d")
    return as_of_date.strftime("%Y%m")


def _parse_ecos_time(value: str, cycle: str) -> date:
    if cycle == "D":
        return datetime.strptime(value, "%Y%m%d").date()
    if cycle == "M":
        parsed = datetime.strptime(value, "%Y%m").date()
        last_day = calendar.monthrange(parsed.year, parsed.month)[1]
        return date(parsed.year, parsed.month, last_day)
    if cycle == "Q":
        match = re.match(r"^(\d{4})Q([1-4])$", value)
        if not match:
            raise ValueError(f"Unsupported ECOS quarter value: {value}")
        year = int(match.group(1))
        quarter = int(match.group(2))
        month = quarter * 3
        return date(year, month, calendar.monthrange(year, month)[1])
    return datetime.strptime(value, "%Y").date()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in {"", "."}:
        return None
    try:
        parsed = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _macro_cache_dir() -> Path:
    path = Path(get_config()["data_cache_dir"]) / "macro"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_cache_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _cache_file(
    source: str,
    series_id: str,
    market: str,
    as_of_date: str,
    lookback_months: int,
) -> Path:
    filename = "__".join(
        [
            _safe_cache_token(source.lower()),
            _safe_cache_token(series_id),
            _safe_cache_token(market),
            _safe_cache_token(as_of_date),
            str(lookback_months),
        ]
    )
    return _macro_cache_dir() / f"{filename}.json"


def _read_cache(
    source: str,
    series_id: str,
    market: str,
    as_of_date: str,
    lookback_months: int,
) -> dict[str, Any] | None:
    path = _cache_file(source, series_id, market, as_of_date, lookback_months)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(
    source: str,
    series_id: str,
    market: str,
    as_of_date: str,
    lookback_months: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    cache_payload = {
        "cache_created_at": _utc_now_iso(),
        "source": source,
        "series_id": series_id,
        "market": market,
        "as_of_date": as_of_date,
        "lookback_months": lookback_months,
        "payload": payload,
    }
    path = _cache_file(source, series_id, market, as_of_date, lookback_months)
    path.write_text(json.dumps(cache_payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return cache_payload


def _get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def _normalize_observations(
    observations: list[tuple[date, float]],
    as_of_date: date,
) -> list[tuple[date, float]]:
    clean = [(obs_date, value) for obs_date, value in observations if obs_date <= as_of_date]
    clean.sort(key=lambda row: row[0])
    return clean


def _nearest_prior_value(
    observations: list[tuple[date, float]],
    target_date: date,
) -> float | None:
    candidate = None
    for obs_date, value in observations:
        if obs_date <= target_date:
            candidate = value
        else:
            break
    return candidate


def _zscore(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std == 0:
        return None
    return round((values[-1] - mean) / std, 2)


def _quality_for_lag(frequency: str, lag_days: int) -> str:
    if frequency == "daily":
        return "ok" if lag_days <= 10 else "stale"
    if frequency == "monthly":
        return "ok" if lag_days <= 75 else "stale"
    return "ok"


def _series_summary(
    *,
    series_def: dict[str, Any],
    source: str,
    observations: list[tuple[date, float]],
    as_of_date: date,
    cache_status: str,
    cache_created_at: str | None,
    quality_note: str | None = None,
) -> dict[str, Any]:
    if not observations:
        return {
            "id": series_def["id"],
            "name": series_def["name"],
            "source": source,
            "frequency": series_def["frequency"],
            "unit": series_def["unit"],
            "latest_observation_date": None,
            "latest_value": None,
            "change_1m": None,
            "change_3m": None,
            "change_6m": None,
            "zscore_18m": None,
            "data_lag_days": None,
            "quality": "unavailable",
            "quality_note": quality_note or "No valid observations at or before curr_date.",
            "cache_status": cache_status,
            "cache_created_at": cache_created_at,
        }

    latest_date, latest_value = observations[-1]
    values = [value for _, value in observations]

    def delta(months: int) -> float | None:
        target = _subtract_months(latest_date, months)
        if series_def["frequency"] == "monthly":
            target = date(target.year, target.month, calendar.monthrange(target.year, target.month)[1])
        prior = _nearest_prior_value(observations, target)
        if prior is None:
            return None
        return round(latest_value - prior, 4)

    lag_days = (as_of_date - latest_date).days
    quality = _quality_for_lag(series_def["frequency"], lag_days)
    return {
        "id": series_def["id"],
        "name": series_def["name"],
        "source": source,
        "frequency": series_def["frequency"],
        "unit": series_def["unit"],
        "latest_observation_date": latest_date.isoformat(),
        "latest_value": round(latest_value, 4),
        "change_1m": delta(1),
        "change_3m": delta(3),
        "change_6m": delta(6),
        "zscore_18m": _zscore(values),
        "data_lag_days": lag_days,
        "quality": quality,
        "quality_note": quality_note or "",
        "cache_status": cache_status,
        "cache_created_at": cache_created_at,
    }


def _parse_fred_observations(payload: dict[str, Any], as_of_date: date) -> list[tuple[date, float]]:
    rows = payload.get("observations")
    if not isinstance(rows, list):
        raise ValueError("FRED payload missing observations list.")
    observations = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _safe_float(row.get("value"))
        if value is None:
            continue
        obs_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
        observations.append((obs_date, value))
    return _normalize_observations(observations, as_of_date)


def _parse_ecos_observations(
    payload: dict[str, Any],
    as_of_date: date,
    cycle: str,
) -> list[tuple[date, float]]:
    container = payload.get("StatisticSearch")
    if not isinstance(container, dict):
        raise ValueError("ECOS payload missing StatisticSearch object.")
    rows = container.get("row")
    if not isinstance(rows, list):
        result = container.get("RESULT", {})
        message = result.get("MESSAGE") if isinstance(result, dict) else None
        raise ValueError(message or "ECOS payload missing row list.")
    observations = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _safe_float(row.get("DATA_VALUE"))
        if value is None:
            continue
        obs_date = _parse_ecos_time(str(row["TIME"]), cycle)
        observations.append((obs_date, value))
    return _normalize_observations(observations, as_of_date)


def _fetch_fred_series(
    series_def: dict[str, Any],
    market: str,
    as_of_date: date,
    lookback_months: int,
    api_key: str,
) -> dict[str, Any]:
    as_of_text = as_of_date.isoformat()
    cached = _read_cache("FRED", series_def["id"], market, as_of_text, lookback_months)
    cache_status = "hit" if cached else "miss"
    if cached:
        payload = cached["payload"]
        cache_created_at = cached.get("cache_created_at")
    else:
        params = {
            "series_id": series_def["id"],
            "api_key": api_key,
            "file_type": "json",
            "observation_start": _subtract_months(as_of_date, lookback_months + 1).isoformat(),
            "observation_end": as_of_text,
            "sort_order": "asc",
        }
        if as_of_date < _today_utc():
            params["realtime_start"] = as_of_text
            params["realtime_end"] = as_of_text
        payload = _get_json(FRED_API_URL, params=params)
        cache_created_at = _write_cache(
            "FRED", series_def["id"], market, as_of_text, lookback_months, payload
        )["cache_created_at"]

    observations = _parse_fred_observations(payload, as_of_date)
    return _series_summary(
        series_def=series_def,
        source="FRED",
        observations=observations,
        as_of_date=as_of_date,
        cache_status=cache_status,
        cache_created_at=cache_created_at,
    )


def _fetch_ecos_series(
    series_def: dict[str, Any],
    market: str,
    as_of_date: date,
    lookback_months: int,
    api_key: str,
) -> dict[str, Any]:
    as_of_text = as_of_date.isoformat()
    cached = _read_cache("ECOS", series_def["id"], market, as_of_text, lookback_months)
    cache_status = "hit" if cached else "miss"
    if cached:
        payload = cached["payload"]
        cache_created_at = cached.get("cache_created_at")
    else:
        cycle = series_def["cycle"]
        start = _period_start(as_of_date, lookback_months, cycle)
        end = _period_end(as_of_date, cycle)
        url = "/".join(
            [
                ECOS_API_ROOT,
                api_key,
                "json",
                "kr",
                "1",
                "100000",
                series_def["stat_code"],
                cycle,
                start,
                end,
                series_def["item_code"],
            ]
        )
        payload = _get_json(url)
        cache_created_at = _write_cache(
            "ECOS", series_def["id"], market, as_of_text, lookback_months, payload
        )["cache_created_at"]

    observations = _parse_ecos_observations(payload, as_of_date, series_def["cycle"])
    return _series_summary(
        series_def=series_def,
        source="BOK ECOS",
        observations=observations,
        as_of_date=as_of_date,
        cache_status=cache_status,
        cache_created_at=cache_created_at,
    )


def _unavailable_block(
    source: str,
    reason: str,
    required_env: str,
    impact: str,
    quality: str = "low",
    error: str | None = None,
) -> dict[str, Any]:
    block = {
        "available": False,
        "source": source,
        "reason": reason,
        "required_env": required_env,
        "impact": impact,
        "quality": quality,
    }
    if error:
        block["error"] = error
    return block


def _source_quality(series: list[dict[str, Any]]) -> str:
    if not series:
        return "unavailable_api_error"
    if any(item.get("quality") == "ok" for item in series):
        return "available"
    if any(item.get("quality") == "stale" for item in series):
        return "available_stale"
    return "unavailable_api_error"


def _series_by_id(series: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["id"]: item for item in series}


def _pressure_from_value(value: float | None, high: float, moderate: float) -> str:
    if value is None:
        return "unknown"
    if value >= high:
        return "high"
    if value >= moderate:
        return "moderate"
    return "low"


def _pressure_from_zscore(zscore: float | None, high: float = 1.0, moderate: float = 0.4) -> str:
    if zscore is None:
        return "unknown"
    if zscore >= high:
        return "high"
    if zscore >= moderate:
        return "moderate"
    if zscore <= -high:
        return "low"
    return "neutral"


def _derive_metrics(market: str, series: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = _series_by_id(series)

    dgs10 = by_id.get("DGS10", {}).get("latest_value")
    dgs2 = by_id.get("DGS2", {}).get("latest_value")
    us_curve = round(dgs10 - dgs2, 4) if dgs10 is not None and dgs2 is not None else None

    kr10 = by_id.get("817Y002:010210000", {}).get("latest_value")
    kr1 = by_id.get("817Y002:010190000", {}).get("latest_value")
    kr_curve = round(kr10 - kr1, 4) if kr10 is not None and kr1 is not None else None
    yield_curve = us_curve if us_curve is not None else kr_curve

    if market == "KR":
        rate_source = by_id.get("722Y001:0101000") or by_id.get("817Y002:010210000", {})
        inflation_source = by_id.get("901Y009:0", {})
        fx_source = by_id.get("731Y001:0000001", {})
        growth_source = by_id.get("901Y067:I16A") or by_id.get("512Y013:99988", {})
        volatility_source = {}
    else:
        rate_source = by_id.get("FEDFUNDS") or by_id.get("DGS10", {})
        inflation_source = by_id.get("CPIAUCSL", {})
        fx_source = by_id.get("DEXKOUS", {})
        growth_source = by_id.get("UNRATE", {})
        volatility_source = by_id.get("VIXCLS", {})

    rate_pressure = _pressure_from_value(rate_source.get("latest_value"), high=4.5, moderate=3.0)
    inflation_pressure = _pressure_from_zscore(inflation_source.get("zscore_18m"))
    fx_pressure = _pressure_from_zscore(fx_source.get("zscore_18m"))
    volatility_value = volatility_source.get("latest_value")
    if volatility_value is None:
        volatility_regime = "unknown"
    elif volatility_value >= 25:
        volatility_regime = "elevated"
    elif volatility_value >= 18:
        volatility_regime = "watch"
    else:
        volatility_regime = "calm"

    growth_z = growth_source.get("zscore_18m")
    if market == "US" and growth_source.get("id") == "UNRATE":
        growth_risk = _pressure_from_zscore(growth_z)
    elif growth_z is None:
        growth_risk = "unknown"
    elif growth_z <= -1.0:
        growth_risk = "high"
    elif growth_z <= -0.4:
        growth_risk = "moderate"
    else:
        growth_risk = "neutral"

    elevated_flags = {rate_pressure, inflation_pressure, fx_pressure, volatility_regime, growth_risk}
    if "high" in elevated_flags or "elevated" in elevated_flags:
        macro_regime = "risk-sensitive"
    elif "moderate" in elevated_flags or "watch" in elevated_flags:
        macro_regime = "balanced-watch"
    elif not series:
        macro_regime = "unavailable"
    else:
        macro_regime = "supportive"

    return {
        "yield_curve_10y2y": yield_curve,
        "rate_pressure": rate_pressure,
        "inflation_pressure": inflation_pressure,
        "growth_risk": growth_risk,
        "fx_pressure": fx_pressure,
        "volatility_regime": volatility_regime,
        "macro_regime": macro_regime,
    }


def _overall_quality(series: list[dict[str, Any]], unavailable: list[dict[str, Any]]) -> str:
    ok_count = sum(1 for item in series if item.get("quality") == "ok")
    if ok_count >= 3 and not unavailable:
        return "high"
    if ok_count >= 1:
        return "medium"
    return "low"


def _root_cache_status(series: list[dict[str, Any]]) -> str:
    if not series:
        return "skipped_no_fetch"
    statuses = {item.get("cache_status") for item in series}
    if statuses == {"hit"}:
        return "hit"
    if statuses == {"miss"}:
        return "miss"
    return "mixed"


def _empty_snapshot(
    *,
    market: str,
    curr_date: str,
    lookback_months: int,
    source_status: dict[str, str],
    unavailable_sources: list[dict[str, Any]],
    quality_note: str,
) -> dict[str, Any]:
    return {
        "available": False,
        "as_of_date": curr_date,
        "market": market,
        "lookback_months": lookback_months,
        "source_status": source_status,
        "series": [],
        "derived": {
            "yield_curve_10y2y": None,
            "rate_pressure": "unknown",
            "inflation_pressure": "unknown",
            "growth_risk": "unknown",
            "fx_pressure": "unknown",
            "volatility_regime": "unknown",
            "macro_regime": "unavailable",
        },
        "unavailable_sources": unavailable_sources,
        "data_revision_risk": False,
        "quality": "low",
        "quality_note": quality_note,
        "cache_status": "skipped_no_fetch",
        "cache_created_at": _utc_now_iso(),
    }


def build_macro_unavailable_snapshot(
    market: str,
    curr_date: str,
    lookback_months: int = 18,
) -> dict[str, Any]:
    normalized_market = _normalize_market(market)
    source_status = {
        "fred": "not_applicable",
        "ecos": "not_applicable",
    }
    unavailable = []
    if normalized_market in {"US", "GLOBAL"}:
        source_status["fred"] = "unavailable_missing_key"
        unavailable.append(
            _unavailable_block(
                "FRED",
                "missing_api_key",
                "FRED_API_KEY",
                "External FRED macro indicators were not used.",
            )
        )
    if normalized_market == "KR":
        source_status["ecos"] = "unavailable_missing_key"
        unavailable.append(
            _unavailable_block(
                "BOK ECOS",
                "missing_api_key",
                "BOK_ECOS_API_KEY",
                "External ECOS macro indicators were not used.",
            )
        )

    return _empty_snapshot(
        market=normalized_market,
        curr_date=curr_date,
        lookback_months=lookback_months,
        source_status=source_status,
        unavailable_sources=unavailable,
        quality_note="External macro API keys are missing, so no macro indicators were fetched.",
    )


def _fetch_source_series(
    *,
    source: str,
    market: str,
    as_of_date: date,
    lookback_months: int,
    api_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    series_results = []
    errors = []
    registry = FRED_SERIES.get(market, []) if source == "FRED" else ECOS_SERIES.get(market, [])
    fetcher = _fetch_fred_series if source == "FRED" else _fetch_ecos_series
    for series_def in registry:
        try:
            summary = fetcher(series_def, market, as_of_date, lookback_months, api_key)
            if summary.get("quality") != "unavailable":
                series_results.append(summary)
        except Exception as exc:  # noqa: BLE001 - tool must degrade instead of crashing.
            errors.append(
                {
                    "series_id": series_def["id"],
                    "source": source,
                    "error": str(exc),
                }
            )
    return series_results, errors


@tool
def get_macro_indicators(
    market: Annotated[str, "Market code: US, KR, or GLOBAL"],
    curr_date: Annotated[str, "Current analysis date in yyyy-mm-dd format"],
    lookback_months: Annotated[
        int,
        "Number of months to inspect for raw/derived macro indicators",
    ] = 18,
) -> str:
    """
    Retrieve a structured macro indicator snapshot from FRED and/or BOK ECOS.

    API keys are optional. Missing keys, bad API responses, and unsupported
    markets return structured low-quality snapshots instead of raising.
    """
    normalized_market = _normalize_market(market)
    try:
        as_of_date = _parse_as_of_date(curr_date)
    except ValueError:
        snapshot = _empty_snapshot(
            market=normalized_market,
            curr_date=curr_date,
            lookback_months=lookback_months,
            source_status={"fred": "not_applicable", "ecos": "not_applicable"},
            unavailable_sources=[],
            quality_note="curr_date must be in YYYY-MM-DD format.",
        )
        snapshot["unavailable_sources"].append(
            _unavailable_block(
                "macro",
                "invalid_curr_date",
                "",
                "Macro indicators were not fetched.",
                error=f"Invalid curr_date: {curr_date}",
            )
        )
        return json.dumps(snapshot, ensure_ascii=False, sort_keys=True)

    source_status = {
        "fred": "not_applicable",
        "ecos": "not_applicable",
    }
    unavailable_sources = []
    errors = []
    series_results = []

    fred_key = os.getenv("FRED_API_KEY")
    ecos_key = os.getenv("BOK_ECOS_API_KEY")

    if normalized_market in {"US", "GLOBAL"}:
        if fred_key:
            fred_series, fred_errors = _fetch_source_series(
                source="FRED",
                market=normalized_market,
                as_of_date=as_of_date,
                lookback_months=lookback_months,
                api_key=fred_key,
            )
            series_results.extend(fred_series)
            errors.extend(fred_errors)
            source_status["fred"] = _source_quality(fred_series)
            if not fred_series:
                unavailable_sources.append(
                    _unavailable_block(
                        "FRED",
                        "api_error",
                        "FRED_API_KEY",
                        "FRED macro indicators were unavailable.",
                        error="; ".join(error["error"] for error in fred_errors[:3]),
                    )
                )
        else:
            source_status["fred"] = "unavailable_missing_key"
            unavailable_sources.append(
                _unavailable_block(
                    "FRED",
                    "missing_api_key",
                    "FRED_API_KEY",
                    "External FRED macro indicators were not used.",
                )
            )

    if normalized_market == "KR":
        if ecos_key:
            ecos_series, ecos_errors = _fetch_source_series(
                source="ECOS",
                market=normalized_market,
                as_of_date=as_of_date,
                lookback_months=lookback_months,
                api_key=ecos_key,
            )
            series_results.extend(ecos_series)
            errors.extend(ecos_errors)
            source_status["ecos"] = _source_quality(ecos_series)
            if not ecos_series:
                unavailable_sources.append(
                    _unavailable_block(
                        "BOK ECOS",
                        "api_error",
                        "BOK_ECOS_API_KEY",
                        "ECOS macro indicators were unavailable.",
                        error="; ".join(error["error"] for error in ecos_errors[:3]),
                    )
                )
        else:
            source_status["ecos"] = "unavailable_missing_key"
            unavailable_sources.append(
                _unavailable_block(
                    "BOK ECOS",
                    "missing_api_key",
                    "BOK_ECOS_API_KEY",
                    "External ECOS macro indicators were not used.",
                )
            )

    data_revision_risk = normalized_market == "KR" and as_of_date < _today_utc()
    quality = _overall_quality(series_results, unavailable_sources)
    quality_note = ""
    if data_revision_risk:
        quality_note = "Historical ECOS data may reflect currently revised values."
        if quality == "high":
            quality = "medium"
    elif not series_results:
        quality_note = "No external macro indicators were available."

    snapshot = {
        "available": bool(series_results),
        "as_of_date": curr_date,
        "market": normalized_market,
        "lookback_months": lookback_months,
        "source_status": source_status,
        "series": series_results,
        "derived": _derive_metrics(normalized_market, series_results),
        "unavailable_sources": unavailable_sources,
        "errors": errors,
        "data_revision_risk": data_revision_risk,
        "quality": quality,
        "quality_note": quality_note,
        "cache_status": _root_cache_status(series_results),
        "cache_created_at": _utc_now_iso(),
    }
    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
