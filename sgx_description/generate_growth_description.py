from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from datetime import datetime, timedelta

from .generate_description import fetch_data, open_json, write_json
from .generate_pe_description import (
    get_symbol_treatment,
    apply_ticker_links,
    FULLY_EXCLUDED_TREATMENTS,
    AGGREGATE_ONLY_EXCLUDED_TREATMENTS,
)
from sgx_description.llm.client import get_model
from sgx_description.llm.prompt import PromptCollections, GrowthDescription

import time


def compute_growth(current_value, prior_value):
    if current_value is None or prior_value is None or prior_value == 0:
        return None

    return (current_value - prior_value) / prior_value


def run_get_data_growth(sectors: list[str], output_path: str):
    final = {}

    sector_reports = fetch_data(
        table='sgx_sector_report',
        columns='sector, weighted_avg_growth_data',
    )

    sector_reports = {
        record['sector']: record.get('weighted_avg_growth_data') or {}
        for record in sector_reports
    }

    for sector in sectors:
        print(f'processing: {sector}')

        last_year = str((datetime.now() - timedelta(365)).year)
        prior_year = str(int(last_year) - 1)

        latest_growth_entry = sector_reports.get(sector, {}).get(last_year)
        
        avg_annual_earning_growth = (
            latest_growth_entry.get('avg_annual_earning_growth') 
            if latest_growth_entry else None
        )

        avg_annual_revenue_growth = (
            latest_growth_entry.get('avg_annual_revenue_growth') 
            if latest_growth_entry else None
        )

        data_companies = fetch_data(
            table='sgx_company_report',
            columns='symbol, name, historical_financials',
            query=lambda q: q.eq('sector', sector).order('name', desc=False),
        )

        companies = {}

        for record in data_companies:
            symbol = record['symbol']
            company_name = record['name']
            historical_financials = record.get('historical_financials') or {}

            current_entry = historical_financials.get(last_year)
            prior_entry = historical_financials.get(prior_year)

            current_earning = current_entry.get('earnings') if current_entry else None
            prior_earning = prior_entry.get('earnings') if prior_entry else None
            current_revenue = current_entry.get('revenue') if current_entry else None
            prior_revenue = prior_entry.get('revenue') if prior_entry else None

            companies[symbol] = {
                "company_name": company_name,
                "earnings_growth": compute_growth(
                    current_earning, 
                    prior_earning
                ),
                "revenue_growth": compute_growth(
                    current_revenue, 
                    prior_revenue
                ),
            }

        final[sector] = {
            "avg_annual_earning_growth": avg_annual_earning_growth,
            "avg_annual_revenue_growth": avg_annual_revenue_growth,
            "companies": companies,
        }

    write_json(path=output_path, payload=final)


def partition_sector_companies(sector_name: str, sector_data: dict):
    company_records = sector_data.get("companies") or {}

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

    for symbol, growth_values in company_records.items():
        treatment = get_symbol_treatment(symbol)

        if treatment in FULLY_EXCLUDED_TREATMENTS:
            excluded_symbols.append(symbol)
            continue

        if treatment in AGGREGATE_ONLY_EXCLUDED_TREATMENTS:
            aggregate_only_symbols.add(symbol)

        included_companies[symbol] = growth_values

    aggregate_eligible_companies = {
        symbol: growth_values
        for symbol, growth_values in included_companies.items()
        if symbol not in aggregate_only_symbols
    }

    return {
        "avg_annual_earning_growth": sector_data.get("avg_annual_earning_growth"),
        "avg_annual_revenue_growth": sector_data.get("avg_annual_revenue_growth"),
        "included_companies": included_companies,
        "aggregate_eligible_companies": aggregate_eligible_companies,
        "excluded_symbols": excluded_symbols,
    }


def find_top_company(companies: dict, growth_field: str):
    candidates = [
        (symbol, values.get("company_name"), values[growth_field])
        for symbol, values in companies.items()
        if values.get(growth_field) is not None
    ]

    if not candidates:
        return None

    symbol, company_name, _ = max(candidates, key=lambda entry: entry[2])

    return {"symbol": symbol, "company_name": company_name}


def rank_sectors_by_growth(sector_partitions: dict, growth_field: str):
    ranks = {}

    growth_values = {
        sector_name: partition[growth_field]
        for sector_name, partition in sector_partitions.items()
        if partition[growth_field] is not None
    }

    for sector_name, growth_value in growth_values.items():
        higher_count = sum(
            1
            for other_growth in growth_values.values()
            if other_growth > growth_value
        )

        ranks[sector_name] = higher_count + 1

    return ranks


def generate_sector_summaries(path: str, output_path: str):
    raw_sector_data = open_json(path)

    sector_partitions = {}

    for sector_name, sector_data in raw_sector_data.items():
        partition = partition_sector_companies(sector_name, sector_data)

        if partition is not None:
            sector_partitions[sector_name] = partition

    earnings_ranks = rank_sectors_by_growth(sector_partitions, "avg_annual_earning_growth")
    revenue_ranks = rank_sectors_by_growth(sector_partitions, "avg_annual_revenue_growth")

    sector_summaries = []

    for sector_name, partition in sector_partitions.items():
        aggregate_eligible_companies = partition["aggregate_eligible_companies"]
        included_companies = partition["included_companies"]
        excluded_symbols = partition["excluded_symbols"]

        top_earning = find_top_company(aggregate_eligible_companies, "earnings_growth")
        top_revenue = find_top_company(aggregate_eligible_companies, "revenue_growth")

        sector_summaries.append({
            "sector_name": sector_name,
            "company_count": len(included_companies),
            "avg_annual_earning_growth": partition["avg_annual_earning_growth"],
            "avg_annual_revenue_growth": partition["avg_annual_revenue_growth"],
            "earnings_rank": earnings_ranks.get(sector_name),
            "revenue_rank": revenue_ranks.get(sector_name),
            "total_sectors": len(earnings_ranks),
            "top_earning": top_earning,
            "top_revenue": top_revenue,
            "excluded_symbols": excluded_symbols,
        })

    write_json(output_path, sector_summaries)


def run_generate_description(sector_summaries_path: str, output_path: str):
    sector_summaries = open_json(sector_summaries_path)

    prompts = PromptCollections()

    system_prompt = prompts.system_prompt()
    user_prompt = prompts.user_prompt_growth()

    model = get_model(model_name='openai/gpt-oss-120b')

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ('user', user_prompt)
    ])

    scoring_parser = JsonOutputParser(pydantic_object=GrowthDescription)
    format_instructions = scoring_parser.get_format_instructions()

    chain = prompt | model | scoring_parser

    growth_descriptions = {}

    for sector_summary in sector_summaries:
        sector_name = sector_summary["sector_name"]
        avg_annual_earning_growth = sector_summary["avg_annual_earning_growth"]
        avg_annual_revenue_growth = sector_summary["avg_annual_revenue_growth"]

        if avg_annual_earning_growth is None and avg_annual_revenue_growth is None:
            print(
                f"Skipping prompt generation for {sector_name}: no valid growth data.",
                sector_name,
            )
            continue

        top_earning = sector_summary["top_earning"]
        top_revenue = sector_summary["top_revenue"]

        named_companies = [
            entry 
            for entry in (top_earning, top_revenue) 
            if entry is not None
        ]

        input_data = {
            "sector": sector_name,
            "avg_annual_earning_growth": (
                round(avg_annual_earning_growth * 100, 2)
                if avg_annual_earning_growth is not None else "unavailable"
            ),
            "avg_annual_revenue_growth": (
                round(avg_annual_revenue_growth * 100, 2)
                if avg_annual_revenue_growth is not None else "unavailable"
            ),
            "earnings_rank": sector_summary["earnings_rank"],
            "revenue_rank": sector_summary["revenue_rank"],
            "total_sectors": sector_summary["total_sectors"],
            "top_earning": top_earning["company_name"] if top_earning else "unavailable",
            "top_revenue": top_revenue["company_name"] if top_revenue else "unavailable",
            "format_instructions": format_instructions,
        }

        result = chain.invoke(input_data)

        print(f'reasoning description for sector: {sector_name}: {result.get("reasoning")}')

        description = result.get('description')

        clean_description = apply_ticker_links(
            description_text=description,
            undervalued_companies=named_companies,
        )

        print(f'\ndescription: {clean_description}\n')

        growth_descriptions[sector_name] = clean_description

        time.sleep(3)

    write_json(output_path, growth_descriptions)


if __name__ == "__main__":
    sectors = open_json('sgx_description/data/sector_list.json')

    sectors = [
        record['sector']
        for record in sectors
    ]

    base_dir = 'sgx_description/data/growth_description/'

    run_get_data_growth(sectors, f'{base_dir}data_for_generation.json')

    generate_sector_summaries(f'{base_dir}data_for_generation.json', f'{base_dir}sector_summaries.json')

    run_generate_description(f'{base_dir}sector_summaries.json', f'{base_dir}generated_description.json')


# uv run -m sgx_description.generate_growth_description
