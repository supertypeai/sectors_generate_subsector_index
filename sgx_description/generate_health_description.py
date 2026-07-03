from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from datetime import datetime 

from statistics import median

from .symbol_exceptions import SGX_SYMBOL_TREATMENT
from .generate_description import (
    fetch_metric_yfinance, 
    fetch_daily_data, 
    fetch_data,
    open_json, 
    write_json
)
from sgx_description.llm.client import get_model
from sgx_description.llm.prompt import PromptCollections, HealthDescription

import time 


MINIMUM_VALID_PRICE = 0.01

IDENTITY_EXCLUSION_TREATMENTS = frozenset({"exclude", "exclude_from_aggregate_only"})
 
IDENTITY_DUPLICATE_REASONS = frozenset({
    "dual_currency_secondary",
    "phantom_scraper_entry",
    "rights_or_warrants_counter",
})
 

def is_excluded_from_aggregation(symbol: str) -> tuple[bool, str | None]:
    entry = SGX_SYMBOL_TREATMENT.get(symbol)
 
    if entry is None:
        return False, None
 
    treatment = entry.get("treatment")
 
    if treatment in IDENTITY_EXCLUSION_TREATMENTS:
        return True, entry.get("reason", treatment)
 
    return False, None


def fallback_fetch_price(records: dict):
    for symbol, item in records.items():
        close_today = item['close_today']
        close_30d = item['close_30d']

        if close_today is not None and close_30d is not None:
            continue

        result = fetch_metric_yfinance(
            symbol=f'{symbol}.SI',
            metric='price',
            days=30        
        )  

        if result is None:
            continue

        today_close_yfinance, close_30d_yfinance = result

        if close_today is None:
            item['close_today'] = today_close_yfinance

        if close_30d is None:
            item['close_30d'] = close_30d_yfinance


def is_valid_price(price: float | None) -> bool:
    return price is not None and price >= MINIMUM_VALID_PRICE


def calculate_return(close: float | None, close_30d: float | None):
    if close is None or close_30d is None:
        return None

    if close_30d <= 0:
        print(f'skipping return: invalid base price close_30d={close_30d}')
        return None

    return round(((close - close_30d) / close_30d) * 100, 2)


def run_get_data_health_description(sectors: list[str], path: str):
    final_data = []

    for sector in sectors:
        print(f'processing {sector}')

        initial_data = fetch_data(
            table='sgx_company_report',
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

        daily_records = fetch_daily_data(symbols, 30)

        fallback_fetch_price(daily_records)

        sector_returns = []

        for symbol, record in daily_records.items():
            close_today = record.get('close_today')
            close_30d = record.get('close_30d')

            if is_valid_price(close_today) and is_valid_price(close_30d):
                computed_return = calculate_return(close_today, close_30d)

            else:
                computed_return = None
 
            record["return_30d"] = computed_return
 
            excluded, exclusion_reason = is_excluded_from_aggregation(symbol)
            record["excluded_from_aggregate"] = excluded
            record["exclusion_reason"] = exclusion_reason
 
            if computed_return is not None and not excluded:
                sector_returns.append(computed_return)

        sector_median = round(median(sector_returns), 2) if sector_returns else None
 
        daily_records['sector_median'] = sector_median

        final_data.append({
            sector: daily_records
        })

    write_json(path, final_data)


def calculate_sgx_overall_avg(path: str):
    records = open_json(path)

    all_returns = []

    for record in records:
        for value in record.values():
            for sub_value in value.values():
                if not isinstance(sub_value, dict):
                    continue

                return_30d = sub_value.get('return_30d')

                if return_30d is None:
                    continue

                all_returns.append(return_30d)

    all_returns.sort()

    print('lowest:', all_returns[:5])
    print('highest:', all_returns[-5:])
 
    overall_median = round(median(all_returns), 2) if all_returns else None
    
    return overall_median

 
def categorize_company(company_data):
    exclusion_reason = company_data.get("exclusion_reason")
 
    if exclusion_reason in IDENTITY_DUPLICATE_REASONS:
        return "not_distinct_company"
 
    if company_data.get("excluded_from_aggregate"):
        return "excluded_from_comparison"
 
    return_30d = company_data.get("return_30d")
 
    if return_30d is None:
        return "no_reliable_price_data"
 
    if return_30d > 0:
        return "positive"
 
    if return_30d < 0:
        return "negative"
 
    return "unchanged"
 
 
def summarize_sector(sector, companies):
    category_counts = {
        "positive": 0,
        "negative": 0,
        "unchanged": 0,
        "no_reliable_price_data": 0,
        "excluded_from_comparison": 0,
        "not_distinct_company": 0,
    }
 
    for symbol, company_data in companies.items():
        if symbol == "sector_median" or not isinstance(company_data, dict):
            continue
 
        category = categorize_company(company_data)
        category_counts[category] += 1
 
    total_companies = sum(
        count
        for category, count in category_counts.items()
        if category != "not_distinct_company"
    )
 
    valid_return_count = (
        category_counts["positive"]
        + category_counts["negative"]
        + category_counts["unchanged"]
    )
 
    return {
        "sector": sector,
        "total_companies": total_companies,
        "valid_return_count": valid_return_count,
        "positive_count": category_counts["positive"],
        "negative_count": category_counts["negative"],
        "unchanged_count": category_counts["unchanged"],
        "no_reliable_price_data_count": category_counts["no_reliable_price_data"],
        "excluded_from_comparison_count": category_counts["excluded_from_comparison"],
        "sector_median": companies.get("sector_median"),
    }
 
 
def rank_sectors(sector_summaries):
    for sector_summary in sector_summaries:
        current_median = sector_summary["sector_median"]
 
        if current_median is None:
            sector_summary["sector_rank"] = None
            continue
 
        higher_median_count = sum(
            1
            for other_summary in sector_summaries
            if other_summary["sector_median"] is not None
            and other_summary["sector_median"] > current_median
        )
 
        sector_summary["sector_rank"] = higher_median_count + 1
 
    return sector_summaries
 
 
def calculate_overall_pooled_median(records):
    all_returns = []
 
    for record in records:
        for companies in record.values():
            for symbol, company_data in companies.items():
                if symbol == "sector_median" or not isinstance(company_data, dict):
                    continue
 
                category = categorize_company(company_data)
 
                if category in ("positive", "negative", "unchanged"):
                    all_returns.append(company_data["return_30d"])
 
    return round(median(all_returns), 2) if all_returns else None
 
 
def generate_sector_summaries(input_data_path: str, output_path: str):
    records = open_json(input_data_path)
 
    overall_median = calculate_overall_pooled_median(records)
 
    sector_summaries = []
 
    for record in records:
        for sector, companies in record.items():
            sector_summary = summarize_sector(sector, companies)
            sector_summaries.append(sector_summary)
 
    sector_summaries = rank_sectors(sector_summaries)
 
    total_sectors = len(sector_summaries)
 
    for sector_summary in sector_summaries:
        sector_median = sector_summary["sector_median"]
 
        if sector_median is None or overall_median is None:
            comparison_to_overall = None

        elif sector_median > overall_median:
            comparison_to_overall = "outperformed"
        
        elif sector_median < overall_median:
            comparison_to_overall = "underperformed"
        
        else:
            comparison_to_overall = "matched"
 
        sector_summary["comparison_to_overall"] = comparison_to_overall
        sector_summary["overall_median"] = overall_median
        sector_summary["total_sectors"] = total_sectors
 
    write_json(output_path, sector_summaries)


def build_health_comparison_signal(comparison_to_overall):
    if comparison_to_overall == "outperformed":
        comparison_signal = (
            "The sector outperformed SGX overall for the median price change "
            "in the last 30 days."
        )
    
    elif comparison_to_overall == "underperformed":
        comparison_signal = (
            "The sector's median price change is worse than the SGX overall "
            "median price change in the last 30 days."
        )

    else:
        raise ValueError(
            f"Unhandled comparison_to_overall value: {comparison_to_overall}"
        )
    
    return comparison_signal


def run_generate_description(sector_summaries_path: str, output_path: str):
    sector_summaries = open_json(sector_summaries_path)
    
    prompts = PromptCollections()

    system_prompt = prompts.system_prompt()
    user_prompt = prompts.user_prompt_health()

    model = get_model(model_name='openai/gpt-oss-120b')

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ('user', user_prompt )
    ])

    scoring_parser = JsonOutputParser(pydantic_object=HealthDescription)
    format_instructions = scoring_parser.get_format_instructions()
    
    chain = prompt | model | scoring_parser

    health_descriptions = {}

    for sector_summary in sector_summaries:
        sector_name = sector_summary["sector"]

        # temp
        if sector_name == 'Consumer Cyclical':
            sector_name = 'Consumer Cyclicals'

        comparison_signal = build_health_comparison_signal(
            sector_summary["comparison_to_overall"]
        )
    
        input_data = {
            "format_instructions": format_instructions,
            "sector": sector_name,
            "total_companies": sector_summary["total_companies"],
            "valid_return_count": sector_summary["valid_return_count"],
            "positive_price_change_count": sector_summary["positive_count"],
            "sector_rank": sector_summary["sector_rank"],
            "total_sectors": sector_summary["total_sectors"],
            "comparison_signal": comparison_signal,
            "year": datetime.now().year
        }

        result = chain.invoke(input_data)

        print(result['reasoning'])
        print(f'\ndescription: {result['description']}\n')

        health_descriptions[sector_name] = result["description"]

        time.sleep(3)

    write_json(output_path, health_descriptions)


if __name__ == '__main__':
    sectors = open_json('sgx_description/data/sector_list.json')
    
    sectors_list = [
        record['sector']
        for record in sectors
    ]

    output_data_health = 'sgx_description/data/health_description/data_for_generation2.json'

    run_get_data_health_description(
        path=output_data_health,
        sectors=sectors_list
    )

    generate_sector_summaries(
        output_data_health, 
        'sgx_description/data/health_description/sector_summaries.json'
    )

    run_generate_description(
        sector_summaries_path='sgx_description/data/health_description/sector_summaries.json',
        output_path='sgx_description/data/health_description/generated_description.json'
    )


# uv run -m sgx_description.generate_health_description