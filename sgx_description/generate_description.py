from datetime import datetime, timedelta
from pathlib import Path 

from .symbol_exceptions import SGX_SYMBOL_TREATMENT
from sgx_description.utils.config import SUPABASE_CLIENT

import yfinance as yf
import json  


TICKER_USES_BACKSLASH = True

FLAT_RELATIVE_TOLERANCE = 1e-6
 
NOTABLE_COMPANY_LIMIT = 5


def open_json(path: str):
    path = Path(path)

    with path.open('r') as file:
        data = json.load(file)

    return data


def write_json(path: str, payload: list):
    path = Path(path)

    with path.open('w') as file:
        json.dump(payload, file, indent=2)

    print(f'saved to {path}')


def fetch_data(
    table: str = 'sgx_company_report',
    columns: str = 'symbol, name, market_cap',
    query=None,
) -> list:
    base_query = (
        SUPABASE_CLIENT
        .table(table)
        .select(columns)
    )

    if query is not None:
        base_query = query(base_query)

    response = base_query.execute()

    return response.data


def fetch_daily_data(symbols: list[str], target_day: int = 365) -> dict:
    today = datetime.now()
    target_date = today - timedelta(days=target_day)

    start_date = target_date - timedelta(days=3)
    start_date_string = start_date.strftime('%Y-%m-%d')

    response = (
        SUPABASE_CLIENT
        .table('sgx_daily_data')
        .select('symbol, market_cap, date, close')
        .in_('symbol', symbols)
        .gte('date', start_date_string)
        .order('date', desc=False)
        .execute()
    )

    data = response.data

    anchor_by_symbol = {}
    latest_by_symbol = {}

    for record in data:
        symbol = record['symbol']

        if symbol not in anchor_by_symbol:
            anchor_by_symbol[symbol] = record

        latest_by_symbol[symbol] = record

    final_data = {}

    for symbol in symbols:
        anchor_record = anchor_by_symbol.get(symbol)
        latest_record = latest_by_symbol.get(symbol)

        final_data[symbol] = {
            'market_cap': anchor_record['market_cap'] if anchor_record else None,
            'close_30d': anchor_record['close'] if anchor_record else None,
            'close_today': latest_record['close'] if latest_record else None,
        }

    return final_data


def fetch_metric_yfinance(
    symbol: str,
    metric: str = "price",
    days: int = 30,
) -> tuple[float, float] | None:
    today = datetime.now()

    # Add a buffer for weekends and holidays
    window_start = today - timedelta(days=days + 30)

    price_frame = yf.download(
        symbol,
        start=window_start.strftime("%Y-%m-%d"),
        end=(today + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )

    if price_frame.empty:
        print(f"yfinance returned empty for {symbol}")
        return None, None 

    close_series = price_frame["Close"]

    if hasattr(close_series, "columns"):
        close_series = close_series.iloc[:, 0]

    latest_close = float(close_series.iloc[-1])

    target_date = today - timedelta(days=days)
    nearest_index = abs(price_frame.index - target_date).argmin()
    historical_close = float(close_series.iloc[nearest_index])

    if metric == "price":
        return latest_close, historical_close

    if metric == "market_cap":
        ticker = yf.Ticker(symbol)

        market_cap = ticker.fast_info.get("market_cap")

        if market_cap is None:
            market_cap = ticker.info.get("marketCap")

        if market_cap is None:
            print(f"Market cap unavailable for {symbol}")
            return None, None 

        historical_market_cap = (
            market_cap * historical_close / latest_close
        )

        return float(market_cap), float(historical_market_cap)

    return None, None  


def run_get_data_description(sectors: list[str], path: str):
    final_data = {}

    for sector in sectors:
        print(f'processing: {sector}')

        initial_data = fetch_data(
            query=lambda q: (
                q.eq('sector', sector)
                .neq('sub_sector', 'Unknown')
                .order('name', desc=False)
            ),
        )

        symbols = [
            record['symbol']
            for record in initial_data
        ]

        historical_mcap = fetch_daily_data(symbols, target_day=365)

        for record in initial_data:
            symbol = record['symbol']
            current_mcap = record['market_cap']

            if symbol not in historical_mcap:
                print(f'skipping symbol: {symbol}')
                continue

            historic_mcap = historical_mcap[symbol].get('market_cap')

            if historic_mcap is None or current_mcap is None:
                current_mcap_yf, historical_mcap_yf = fetch_metric_yfinance(
                    symbol=f'{symbol}.SI',
                    metric='market_cap',
                    days=365
                )

                if current_mcap is None:
                    record['market_cap'] = current_mcap_yf

                if historic_mcap is None:
                    record['last_365_mcap'] = historical_mcap_yf
                
                else:
                    record['last_365_mcap'] = historic_mcap
            else:
                record['last_365_mcap'] = historic_mcap

        final_data[sector] = initial_data

    write_json(
        path,
        final_data
    )


def apply_symbol_treatment(record: dict) -> dict:
    """
    Return a copy of the record with any last_365_mcap override applied, plus a
    resolved treatment tag. Does not drop anything; dropping is handled by the
    caller so that the reasons stay visible.
    """
    symbol = record["symbol"]
    treatment_entry = SGX_SYMBOL_TREATMENT.get(symbol)
 
    resolved = dict(record)
    resolved["treatment"] = treatment_entry["treatment"] if treatment_entry else "include"
    
    # if treatment_entry and treatment_entry["treatment"] == "override_last_365":
    #     resolved["last_365_mcap"] = treatment_entry["last_365_mcap"]
    #     resolved["treatment"] = "include"

    return resolved
 
 
def partition_records(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split a sector's records into three groups after applying treatments:
      - aggregate_records: counted in the aggregate percentage and pos/neg counts
      - notable_only_records: excluded from aggregate/counts but eligible for the
        notable-companies ranking (foreign secondary listings)
      - dropped_records: excluded everywhere (dual-currency secondaries and any
        record with a null market_cap or last_365_mcap that cannot be recovered)
    """
    aggregate_records = []
    notable_only_records = []
    dropped_records = []
 
    for record in records:
        resolved = apply_symbol_treatment(record)
 
        if resolved["treatment"] == "exclude":
            dropped_records.append(resolved)
            continue
 
        if resolved["market_cap"] is None or resolved["last_365_mcap"] is None:
            dropped_records.append(resolved)
            continue
 
        if resolved["treatment"] == "exclude_from_aggregate":
            notable_only_records.append(resolved)
            continue
 
        aggregate_records.append(resolved)
 
    return aggregate_records, notable_only_records, dropped_records
 
 
def classify_change(current_mcap: float, prior_mcap: float) -> str:
    change_fraction = (current_mcap - prior_mcap) / prior_mcap
 
    if abs(change_fraction) <= FLAT_RELATIVE_TOLERANCE:
        return "flat"
    
    if change_fraction > 0:
        return "positive"
    
    return "negative"
 
 
def compute_sector_metrics(aggregate_records: list[dict], notable_pool: list[dict]) -> dict:
    total_current = sum(record["market_cap"] for record in aggregate_records)
    total_prior = sum(record["last_365_mcap"] for record in aggregate_records)
 
    if total_prior > 0:
        total_change_pct = round((total_current - total_prior) / total_prior * 100, 2)
    
    else:
        total_change_pct = 0.0
 
    positive_count = 0
    negative_count = 0

    for record in aggregate_records:
        classification = classify_change(record["market_cap"], record["last_365_mcap"])
        
        if classification == "positive":
            positive_count += 1
        
        elif classification == "negative":
            negative_count += 1
 
    ranked_by_mcap = sorted(notable_pool, key=lambda record: record["market_cap"], reverse=True)
    notable_companies = ranked_by_mcap[:NOTABLE_COMPANY_LIMIT]
 
    return {
        "total_change_pct": total_change_pct,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "notable_companies": notable_companies,
        "aggregate_company_count": len(aggregate_records),
    }
 
 
def format_ticker_link(symbol: str, name: str) -> str:
    clean_name = name.rstrip(".")

    if TICKER_USES_BACKSLASH:
        return f"[#{symbol}]{clean_name}[\\#{symbol}]"
    
    return f"[#{symbol}]{clean_name}[#{symbol}]"
 
 
def build_trend_sentence(total_change_pct: float, positive_count: int, negative_count: int) -> str:
    if total_change_pct < 0:
        if positive_count < negative_count:
            return (
                f"At this moment, this sector appears to be in a bearish market sentiment, "
                f"with a total market cap depreciation of approximately {total_change_pct}% "
                f"over the past 365 days. It's possible that around {negative_count} companies "
                f"within this sector may have encountered a negative market cap change, while it "
                f"seems that only about {positive_count} companies might have experienced a "
                f"positive market cap change, indicating a challenging market environment."
            )
        return (
            f"Currently, this sector appears to be amidst a bearish trend, marked by an overall "
            f"market capitalization depreciation of approximately {total_change_pct}% over the "
            f"past 365 days. However, upon closer examination of individual companies, it's "
            f"evident that more companies are undergoing positive changes ({positive_count}) in "
            f"market capitalization compared to those experiencing negative changes "
            f"({negative_count}). This suggests that larger companies with negative changes are "
            f"exerting significant influence despite more companies having upward trends."
        )
    
    if total_change_pct > 0:
        if positive_count < negative_count:
            return (
                f"Currently, this sector appears to be on an upward trend, with a notable total "
                f"market capitalization growth of approximately {total_change_pct}% over the last "
                f"365 days. However, a closer look at individual companies reveals that a larger "
                f"number are showing negative changes ({negative_count}) in market capitalization "
                f"compared to those with positive changes ({positive_count}). This indicates that "
                f"the larger companies with positive changes are exerting a significant influence "
                f"on the overall market capitalization, despite more companies experiencing declines."
            )
        
        return (
            f"At this moment, this sector seems to be experiencing a bullish trend, with a total "
            f"market cap growth of around {total_change_pct}% over the past 365 days. It appears "
            f"that {positive_count} companies in this sector might have seen a positive market cap "
            f"change, while around {negative_count} companies could have experienced a negative "
            f"market cap change."
        )
    
    return (
        f"Currently, this sector appears to be in a neutral market sentiment, with no significant "
        f"change in total market capitalization over the past 365 days. It seems that "
        f"approximately {positive_count} companies within this sector may have encountered a "
        f"positive market cap change, while it's possible that around {negative_count} companies "
        f"might have experienced a negative market cap change."
    )
 
 
def build_notable_sentence(notable_companies: list[dict]) -> str | None:
    links = [
        format_ticker_link(record["symbol"], record["name"])
        for record in notable_companies
    ]
 
    if len(links) == 0:
        return None
    
    if len(links) == 1:
        return f"The notable company in this sector, based on market capitalization, is {links[0]}."
    
    if len(links) == 2:
        return (
            f"Notable companies in this sector, based on market capitalization, include "
            f"{links[0]} and {links[1]}."
        )
    
    return (
        f"Notable companies in this sector, based on market capitalization, include "
        f"{', '.join(links[:-1])}, and {links[-1]}."
    )
 
 
def generate_complete_description(sector_overview: str, metrics: dict) -> str:
    trend_sentence = build_trend_sentence(
        metrics["total_change_pct"],
        metrics["positive_count"],
        metrics["negative_count"],
    )

    notable_sentence = build_notable_sentence(metrics["notable_companies"])
 
    parts = [sector_overview.strip(), trend_sentence]

    if notable_sentence is not None:
        parts.append(notable_sentence)
 
    return " ".join(parts)
 
 
def run_generate_descriptions(
    generation_data_path: str,
    sector_overview_path: str,
    output_path: str,
) -> dict:
    sectors = open_json(generation_data_path)
    sector_overviews = open_json(sector_overview_path)
 
    descriptions = {}
    
    for sector, records in sectors.items():
        aggregate_records, notable_only_records, dropped_records = partition_records(records)
        notable_pool = aggregate_records + notable_only_records
        metrics = compute_sector_metrics(aggregate_records, notable_pool)
 
        print(
            f"{sector}: change={metrics['total_change_pct']}% "
            f"positive={metrics['positive_count']} negative={metrics['negative_count']} "
            f"aggregate_companies={metrics['aggregate_company_count']} "
            f"dropped={len(dropped_records)}"
        )
 
        overview = sector_overviews.get(sector)
        
        if overview is None:
            print(
                f"No sector overview found for '{sector}'. Skipping description generation "
                f"for this sector so it is not written with missing static text."
            )
            continue
 
        descriptions[sector] = generate_complete_description(overview, metrics)

    write_json(output_path, descriptions)


if __name__ == '__main__':
    sectors = open_json('sgx_description/data/sector_list.json')
    
    sectors_list = [
        record['sector']
        for record in sectors
    ]

    run_get_data_description(
        path='sgx_description/data/description/data_for_generation.json',
        sectors=sectors_list
    )

    run_generate_descriptions(
        generation_data_path="sgx_description/data/description/data_for_generation.json",
        sector_overview_path="sgx_description/data/description/sector_description.json",
        output_path="sgx_description/data/description/generated_descriptions.json",
    )


# uv run -m sgx_description.generate_description