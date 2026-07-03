from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from datetime import datetime, timedelta
from statistics import median

from .generate_description import fetch_daily_data, fetch_metric_yfinance, fetch_data, open_json, write_json
from .symbol_exceptions import SGX_SYMBOL_TREATMENT
from sgx_description.llm.client import get_model
from sgx_description.llm.prompt import PromptCollections, PEDescription

import time 


FULLY_EXCLUDED_TREATMENTS = {"exclude"}
AGGREGATE_ONLY_EXCLUDED_TREATMENTS = {"exclude_from_aggregate_only"}
 

def run_get_data_pe(sectors: list, output_path: str):
    final = {}

    for sector in sectors:
        print(f'processing: {sector}')

        data_companies = fetch_data(
            table='sgx_company_report',
            columns='symbol, name, pe, earnings, historical_financials, market_cap',
            query=lambda q: q.eq('sector', sector).order('name', desc=False),
        )

        symbols = [record['symbol'] for record in data_companies]
        daily_data = fetch_daily_data(symbols, target_day=365)

        items = {}

        for record in data_companies:
            symbol = record['symbol']
            current_pe = record.get('pe')
            historical_financials = record.get('historical_financials') or {}
            company_name = record['name']

            # Last year
            last_year_mcap = daily_data.get(symbol, {}).get('market_cap')
            last_year = datetime.now() - timedelta(365)

            last_year_entry = historical_financials.get(str(last_year.year))
            last_year_earning = last_year_entry.get('earnings') if last_year_entry else None

            last_year_pe = (
                last_year_mcap / last_year_earning
                if last_year_mcap and last_year_earning else None
            )

            if (current_pe is not None and current_pe > 50) or (last_year_pe is not None and last_year_pe > 50):
                _, last_year_mcap_yf = fetch_metric_yfinance(
                    symbol=f'{symbol}.SI', 
                    metric='market_cap',
                    days=365
                )

                if last_year_mcap_yf is not None and last_year_earning is not None:
                    last_year_pe = (
                        last_year_mcap_yf / last_year_earning
                    )

            items[symbol] = {
                "company_name": company_name,
                "last_year_pe": last_year_pe,
                "current_pe": current_pe
            }

        final[sector] = {
            **items
        }

    write_json(path=output_path, payload=final)


def get_symbol_treatment(symbol):
    treatment_entry = SGX_SYMBOL_TREATMENT.get(symbol)

    if treatment_entry is None:
        return None
    
    return treatment_entry.get("treatment")
 
 
def compute_median_pe(pe_records, pe_field):
    valid_pe_values = [
        pe_record[pe_field]
        for pe_record in pe_records
        if pe_record.get(pe_field) is not None
    ]

    if not valid_pe_values:
        return None
    
    return median(valid_pe_values)
 
 
def determine_pe_status(current_pe_median, last_year_pe_median):
    if current_pe_median is None:
        return "unknown"
    
    if current_pe_median < 0:
        return "negative"
    
    if last_year_pe_median is not None and current_pe_median < last_year_pe_median:
        return "lower"
    
    return "higher"
 
 
def partition_sector_companies(sector_name, sector_data):
    company_records = {
        symbol: pe_values
        for symbol, pe_values in sector_data.items()
        if symbol != "pe_median"
    }
 
    if not company_records:
        print(
            f"Skipping sector {sector_name}: no company records found, likely a "
            "stale or duplicate taxonomy key at the source.",
            sector_name,
        )
        return None
 
    included_companies = {}
    excluded_symbols = []
    aggregate_only_symbols = set()
 
    for symbol, pe_values in company_records.items():
        treatment = get_symbol_treatment(symbol)

        if treatment in FULLY_EXCLUDED_TREATMENTS:
            excluded_symbols.append(symbol)
            continue

        if treatment in AGGREGATE_ONLY_EXCLUDED_TREATMENTS:
            aggregate_only_symbols.add(symbol)

        included_companies[symbol] = pe_values
 
    aggregate_eligible_companies = {
        symbol: pe_values
        for symbol, pe_values in included_companies.items()
        if symbol not in aggregate_only_symbols
    }
 
    return {
        "included_companies": included_companies,
        "aggregate_eligible_companies": aggregate_eligible_companies,
        "excluded_symbols": excluded_symbols,
        "source_sector_pe_median_raw": sector_data.get("pe_median"),
    }
 
 
def generate_sector_summaries(path: str, output_path: str):
    raw_sector_data = open_json(path)

    sector_partitions = {}

    for sector_name, sector_data in raw_sector_data.items():
        partition = partition_sector_companies(sector_name, sector_data)

        if partition is not None:
            sector_partitions[sector_name] = partition
 
    pooled_company_records = []

    for partition in sector_partitions.values():
        pooled_company_records.extend(
            partition["aggregate_eligible_companies"].values()
        )
 
    overall_pe_median = compute_median_pe(
        pooled_company_records, 
        "current_pe"
    )
 
    sector_summaries = []
 
    for sector_name, partition in sector_partitions.items():
        included_companies = partition["included_companies"]
        aggregate_eligible_companies = partition["aggregate_eligible_companies"]
        excluded_symbols = partition["excluded_symbols"]
 
        sector_pe_median = compute_median_pe(
            aggregate_eligible_companies.values(), "current_pe"
        )

        sector_last_year_pe_median = compute_median_pe(
            aggregate_eligible_companies.values(), "last_year_pe"
        )
 
        below_sector_median_count = 0

        if sector_pe_median is not None:
            below_sector_median_count = sum(
                1
                for pe_values in aggregate_eligible_companies.values()
                if pe_values.get("current_pe") is not None
                and 0 < pe_values["current_pe"] < sector_pe_median
            )
 
        below_market_median_count = 0

        if overall_pe_median is not None:
            below_market_median_count = sum(
                1
                for pe_values in aggregate_eligible_companies.values()
                if pe_values.get("current_pe") is not None
                and 0 < pe_values["current_pe"] < overall_pe_median
            )
 
        positive_pe_companies = [
            (symbol, pe_values.get("company_name"), pe_values["current_pe"])
            for symbol, pe_values in included_companies.items()
            if pe_values.get("current_pe") is not None and pe_values["current_pe"] > 0
        ]

        positive_pe_companies.sort(key=lambda company_entry: company_entry[2])
        
        undervalued_companies = [
            {"symbol": symbol, "company_name": company_name}
            for symbol, company_name, _ in positive_pe_companies[:3]
        ]
 
        pe_status = determine_pe_status(sector_pe_median, sector_last_year_pe_median)
 
        sector_summaries.append({
            "sector_name": sector_name,
            "company_count": len(included_companies),
            "sector_pe_median": sector_pe_median,
            "sector_last_year_pe_median": sector_last_year_pe_median,
            "overall_pe_median": overall_pe_median,
            "pe_status": pe_status,
            "below_sector_median_count": below_sector_median_count,
            "below_market_median_count": below_market_median_count,
            "undervalued_companies": undervalued_companies,
            "excluded_symbols": excluded_symbols,
        })
 
    write_json(output_path, sector_summaries)


def format_trend_description(pe_status, sector_last_year_pe_median):
    if pe_status == "unknown" or sector_last_year_pe_median is None:
        return "not comparable to last year, since no reliable prior-year P/E data is available for this sector"
    
    if pe_status == "negative":
        return "reflecting a net loss at the median company"
    
    if pe_status == "lower":
        return f"lower than last year's median of {round(sector_last_year_pe_median, 3)}"
    
    return f"higher than last year's median of {round(sector_last_year_pe_median, 3)}"


def format_ticker_link(symbol: str, company_name: str) -> str:
    return f"[#{symbol}]{company_name}[\\#{symbol}]"


def apply_ticker_links(description_text: str, undervalued_companies: list) -> str:
    formatted_text = description_text
    
    for entry in undervalued_companies:
        ticker_link = format_ticker_link(entry["symbol"], entry["company_name"])
        formatted_text = formatted_text.replace(entry["company_name"], ticker_link)
    
    return formatted_text


def run_generate_description(sector_summaries_path: str, output_path: str):
    sector_summaries = open_json(sector_summaries_path)
    
    prompts = PromptCollections()

    system_prompt = prompts.system_prompt()
    user_prompt = prompts.user_prompt_pe()

    model = get_model(model_name='openai/gpt-oss-120b')

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ('user', user_prompt )
    ])

    scoring_parser = JsonOutputParser(pydantic_object=PEDescription)
    format_instructions = scoring_parser.get_format_instructions()
    
    chain = prompt | model | scoring_parser

    pe_descriptions = {}

    for sector_summary in sector_summaries:
        sector_name = sector_summary["sector_name"]
        sector_pe_median = sector_summary["sector_pe_median"]
    
        if sector_pe_median is None:
            print(
                f"Skipping prompt generation for {sector_name}: no valid sector_pe_median.",
                sector_name,
            )
            continue
    
        overall_pe_median = sector_summary["overall_pe_median"]
        pe_status = sector_summary["pe_status"]
        sector_last_year_pe_median = sector_summary["sector_last_year_pe_median"]
        undervalued_companies = sector_summary["undervalued_companies"]

        trend_description = format_trend_description(pe_status, sector_last_year_pe_median)
        
        undervalued_labels = [
            f"{entry['company_name']} ({entry['symbol']})" if entry.get("company_name") else entry["symbol"]
            for entry in undervalued_companies
        ]
        
        undervalued_text = ", ".join(undervalued_labels) if undervalued_labels else "none identified"
    
        input_data = {
            "sector": sector_name,
            "sector_pe_median": round(sector_pe_median, 3),
            "overall_pe_median": round(overall_pe_median, 3) if overall_pe_median is not None else "unavailable",
            "trend_description": trend_description,
            "company_count": sector_summary["company_count"],
            "below_sector_median_count": sector_summary["below_sector_median_count"],
            "below_market_median_count": sector_summary["below_market_median_count"],
            "undervalued_companies": undervalued_text,
            "format_instructions": format_instructions,
        }
    
        result = chain.invoke(input_data)

        print(f'reasoning description for sector: {sector_name}: {result.get('reasoning')}')

        description = result.get('description')

        clean_description = apply_ticker_links(
            description_text=description, 
            undervalued_companies=undervalued_companies
        )

        print(f'\ndescription: {clean_description}\n')

        pe_descriptions[sector_name] = clean_description

        time.sleep(3)

    write_json(output_path, pe_descriptions)


if __name__ == '__main__':
    sectors = open_json('sgx_description/data/sector_list.json')
    
    sectors = [
        record['sector'] 
        for record in sectors
    ]

    base_dir = 'sgx_description/data/pe_description/'

    run_get_data_pe(sectors, f'{base_dir}data_for_generation.json')

    generate_sector_summaries(f'{base_dir}data_for_generation.json', f'{base_dir}sector_summaries.json')

    run_generate_description(f'{base_dir}sector_summaries.json', f'{base_dir}generated_description.json')


# uv run -m sgx_description.generate_pe_description