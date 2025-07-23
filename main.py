# Import library
import pandas as pd
import json
import numpy as np
import os
import ast
import logging
from supabase import create_client
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

load_dotenv()

# Configure supabase client
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

# Read Data
def read_table_from_supabase(supabase, table_name, columns):
    # Convert list of columns to comma-separated string if needed
    if isinstance(columns, list):
        columns = ",".join(columns)
    
    response = supabase.table(table_name).select(columns).execute()
    response = pd.DataFrame(response.data)
    
    return response

df_compro = read_table_from_supabase(supabase,"idx_company_profile", columns = ["symbol","company_name",'sub_sector_id',"listing_date"])
df_comrep = read_table_from_supabase(supabase,"idx_company_report",columns=["symbol","company_name","sector","sub_sector","historical_valuation","yoy_quarter_earnings_growth","yoy_quarter_revenue_growth"])
df_subsec = read_table_from_supabase(supabase,"idx_subsector_metadata", columns = ["sub_sector_id","sub_sector","sector"])
df_metrics = read_table_from_supabase(supabase,"idx_calc_metrics_daily", columns=["symbol","price_change_30_days","max_drawdown","rsd_close"])
df_sec = read_table_from_supabase(supabase,"idx_sector_reports", columns = ["sector","slug","sub_sector","weighted_avg_growth_data","weighted_max_drawdown","weighted_rsd_close","historical_valuation"])

# Data Processing
def get_last_data_from_json(row, last, metrics):
    data_list = ast.literal_eval(str(row))
    last_metrics_data = data_list[last][metrics]
    return last_metrics_data

def sub_sector_financial_processing(df_sec):
    # Get the last PE value
    df_sec['last_pe_value'] = df_sec['historical_valuation'].apply(lambda x: get_last_data_from_json(x,-1,"pe"))

    # Get the last 2 PE value
    df_sec['last_two_pe_value'] = df_sec['historical_valuation'].apply(lambda x: get_last_data_from_json(x,-2,"pe"))

    # Get avg_annual_earnings_growth and avg_annual_revenue_growth
    df_sec['avg_annual_earnings_growth'] = df_sec['weighted_avg_growth_data'].apply(lambda x: get_last_data_from_json(x,-1,"avg_annual_earning_growth"))
    df_sec['avg_annual_revenue_growth'] = df_sec['weighted_avg_growth_data'].apply(lambda x: get_last_data_from_json(x,-1,"avg_annual_revenue_growth"))

    return df_sec

def price_changes_data(df_metrics):
    # Data sub-sector merging
    df_metrics = df_metrics.merge(df_compro)
    df_metrics = df_metrics.merge(df_subsec)

    # Change symbol format
    format_text = lambda company_name, symbol: f"[#{symbol}]{company_name}[\\#{symbol}]"
    df_metrics["symbol_company"] = df_metrics.apply(lambda row: format_text(row['company_name'], row['symbol']), axis=1)

    # Get average changes per sector
    df_changes_sector = df_metrics.groupby(["sub_sector"]).agg({"price_change_30_days":"mean"}).reset_index()
    df_changes_sector.columns = ["sub_sector","sector_price_change_30_days"]

    df_metrics = df_metrics.merge(df_changes_sector, on =["sub_sector"])

    return df_metrics

def company_pe_processing(df_comrep):
    df_comrep = df_comrep.dropna(subset=["historical_valuation"])

    # Get the last pe_value
    df_comrep['pe_value'] = df_comrep['historical_valuation'].apply(lambda x: get_last_data_from_json(x,-1,"pe"))

    # regenerate symbol
    format_text = lambda company_name, symbol: f"[#{symbol}]{company_name}[\\#{symbol}]"
    df_comrep["symbol_company"] = df_comrep.apply(lambda row: format_text(row['company_name'], row['symbol']), axis=1)

    # Combine with sector PE value data
    df_comrep = df_comrep.merge(df_sec[["sub_sector","last_pe_value"]], on="sub_sector")

    df_comrep.rename(columns={"last_pe_value":"sector_pe_value"}, inplace=True)

    return df_comrep

def new_listing_company(df_compro):
   # Merge company profile data with sub_sector metadata
    df_no_company = df_compro[["symbol",'sub_sector_id',"listing_date"]].merge(df_subsec)

    # Datetime data manipulation
    df_no_company["listing_date"] = pd.to_datetime(df_no_company["listing_date"])
    df_no_company = df_no_company.sort_values("listing_date")
    df_no_company['listing_month'] = df_no_company["listing_date"].dt.month
    df_no_company['listing_year'] = df_no_company["listing_date"].dt.year

    # Calculate number of companies and number of new listing per sub sector
    df_nocom = df_no_company.groupby(["sub_sector_id","sub_sector","sector"]).agg({"symbol":"count","listing_date":"last"}).reset_index()
    df_nocom_this_month = df_no_company[(df_no_company.listing_month == df_no_company.iloc[-1].listing_month) & (df_no_company.listing_year == df_no_company.iloc[-1].listing_year)].groupby(["sub_sector_id","sub_sector","sector"])[["symbol"]].count().reset_index()

    # Merge number of company with the newest new_listing
    df_nocom = df_nocom.merge(df_nocom_this_month[["sub_sector_id","symbol"]], on="sub_sector_id", how="left").fillna(0)

    df_nocom["new_company_percentage"] = round(df_nocom["symbol_y"]/df_nocom["symbol_x"]*100,2)
    
    return df_nocom 

df_sec = sub_sector_financial_processing(df_sec)
df_metrics = price_changes_data(df_metrics)
df_comrep = company_pe_processing(df_comrep)
df_nocom = new_listing_company(df_compro)

# LLM

## Generate P/E ratio index description

### Description generator function
def pe_desc_generator(df_sec,df_comrep,sector):
    # Load the chat model
    model = init_chat_model("gpt-3.5-turbo", model_provider="openai",)

    # Get P/E data from your DataFrames
    subsec_pe = df_sec[df_sec.sub_sector == sector]["last_pe_value"].values[0]

    if subsec_pe < 0:
        status = "negative"
    elif (subsec_pe > 0) & (subsec_pe < df_sec[df_sec.sub_sector == sector]["last_two_pe_value"].values[0]):
        status = "lower"
    else:
        status = "higher"

    pe_rank = df_sec[(df_sec.last_pe_value < subsec_pe) & (df_sec.last_pe_value >= 0)].shape[0] + 1

    sector_comp_num = df_comrep[df_comrep.sub_sector == sector].shape[0]

    outperform_sector = df_comrep[
        (df_comrep.sub_sector == sector) &
        (df_comrep.pe_value < subsec_pe) &
        (df_comrep.pe_value > 0)
    ].shape[0]

    undervalued = df_comrep[
        (df_comrep.pe_value >= 0) &
        (df_comrep.sub_sector == sector)
    ].sort_values("pe_value", ascending=True).head(3)
    undervalued = undervalued.symbol.values

    # === Define the prompt template ===
    prompt_template = PromptTemplate.from_template("""
    Please write a medium-length paragraph that includes some of the following:
    - Describe whether the {sector} sub-sector is considered defensive or cyclical, along with its historical P/E value conditions (moderate, high, or low).
    - Explore potential factors influencing the current P/E value being {status} compared to last year.
    - Mention that {outperform_sector} companies in the {sector} sub-sector have P/E ratios below the market average, with {undervalued} being among the most undervalued.
    - Hypothesize what might be driving this sector’s P/E levels and how it relates to the broader Indonesian economy.

    Avoid giving financial advice and keep it concise.
    """)

    # === Run the chain ===
    chain = prompt_template | model

    result = chain.invoke({
        "sector": sector,
        "status": status,
        "pe_rank": pe_rank,
        "outperform_sector": outperform_sector,
        "undervalued": ', '.join(undervalued),
        "sector_comp_num": sector_comp_num
    })

    result = result.content
    
    for i in undervalued:
        result = result.replace(i, f'[#{i}]{df_comrep[df_comrep.symbol == i].company_name.values[0]}[\#{i}]')

    return result

### Generate description and save it into DB for P/E ratio description
pevalue_index = {}
for i in df_sec.sub_sector:
    try:
        pevalue_index[i] = pe_desc_generator(df_sec,df_comrep,i)
        print(i)
    except:
        logging.error(
            f"Failed to generate complete pe ratio description for {i}. The description on the table won't be updated."
        )

for sub_sector in pevalue_index:
    try:
        test = supabase.table("idx_subsector_metadata").update(
            {"pe_index_description": pevalue_index[sub_sector]}
        ).eq("sub_sector", sub_sector).execute()
        print(test)
    except:
        logging.error(f"Failed to update description for {sub_sector}.")

## Generate Health & Resilience index description

### Description generator function
def health_desc_generator(df_metrics,df_nocom,sector):
    # Initialize Model
    model = init_chat_model("gpt-3.5-turbo", model_provider="openai")

    # IDX average price change
    idx_avg_pricechg = df_metrics.price_change_30_days.mean()
    sector_avg_pricechg = df_metrics[df_metrics.sub_sector == sector].sector_price_change_30_days.values[0]

    if idx_avg_pricechg > sector_avg_pricechg:
        comparison = "The sub-sector's average price changes is worse than the average IDX price changes in the last 30 days"
    else:
        comparison = "The sub-sector outperform IDX for the average price changes in the last 30 days"

    # New listing company
    new_com_pct = df_nocom[df_nocom.sub_sector == sector].new_company_percentage.values[0]

    if new_com_pct > 0:
        status = f"{new_com_pct}% of the company in {sector} is listing this month"
    else:
        date = np.datetime_as_string(df_nocom[df_nocom.sub_sector == sector].listing_date.values[0], unit='D')
        status = f"There aren't any new listing company in this month, the last new listing company in this subsector happened in {date}"

    # Price change count
    sector_nocomp = df_metrics[df_metrics.sub_sector == sector].shape[0]
    pos_price_chg = df_metrics[(df_metrics.sub_sector == sector) & (df_metrics.price_change_30_days > 0)].shape[0]

    # Subsector ranking
    df_metrics_subsec = df_metrics.drop_duplicates('sub_sector')
    subsec_rank = df_metrics_subsec[
        df_metrics_subsec.sector_price_change_30_days >
        df_metrics_subsec[df_metrics_subsec.sub_sector == sector].sector_price_change_30_days.values[0]
    ].shape[0] + 1

    # === Define the prompt ===
    template = """
    Please write a medium-length paragraph that includes some of these: 
    - With {sector_nocomp} companies in the {sector} sub-sector, this sub-sector placed {subsec_rank} in the sub-sector ranking based on the price changes in the previous 30 days.
    - {comparison}
    - Make a hypothesis based on {comparison}, why that could happen for the {sector} sector.
    - In this month, {status}. Add some explanation about barriers to entry based on this month's new company status.
    - From {sector_nocomp} companies, {pos_price_chg} has a positive price change in the last 30 days.
    - Add conditions that can uniquely affect the {sector} market health in the Indonesian index — just mention the factors, no need to elaborate.

    Do not give financial advice. Keep it concise.
    """

    prompt = PromptTemplate.from_template(template)

    # === Chain execution using LCEL ===
    chain = prompt | model

    # Invoke the model
    result = chain.invoke({
        "sector": sector,
        "sector_nocomp": sector_nocomp,
        "subsec_rank": subsec_rank,
        "status": status,
        "comparison": comparison,
        "pos_price_chg": pos_price_chg
    })

    return result.content

### Generate description and save it into DB for Health & Resilience index description
health_index = {}
for i in df_sec.sub_sector:
    try:
        health_index[i] = health_desc_generator(df_metrics,df_nocom,i)
        print(i)
    except:
        logging.error(
            f"Failed to generate complete health & resilience description for {i}. The description on the table won't be updated."
        )

for sub_sector in health_index:
    try:
        supabase.table("idx_subsector_metadata").update(
            {"health_index_description": health_index[sub_sector]}
        ).eq("sub_sector", sub_sector).execute()
    except:
        logging.error(f"Failed to update description for {sub_sector}.")

## Generate Health & Resilience index description
        
### Description generator function
def growth_desc_generator(df_sec,df_comrep,sector):
    # Initialize model
    model = init_chat_model("gpt-3.5-turbo", model_provider="openai")

    # Get the earning and revenue growth of the sub_sector
    subsec_earning_growth = df_sec[df_sec.sub_sector == sector].avg_annual_earnings_growth.values[0]
    subsec_revenue_growth = df_sec[df_sec.sub_sector == sector].avg_annual_revenue_growth.values[0]

    # Rankings
    earn_rank = df_sec[df_sec.avg_annual_earnings_growth > subsec_earning_growth].shape[0] + 1
    rev_rank = df_sec[df_sec.avg_annual_revenue_growth > subsec_revenue_growth].shape[0] + 1

    # Get top earning and revenue companies
    df_comrep_subsec = df_comrep[df_comrep.sub_sector == sector]
    top_earning = df_comrep_subsec[
        df_comrep_subsec.yoy_quarter_earnings_growth == df_comrep_subsec.yoy_quarter_earnings_growth.max()
    ].symbol.values[0]
    top_revenue = df_comrep_subsec[
        df_comrep_subsec.yoy_quarter_revenue_growth == df_comrep_subsec.yoy_quarter_revenue_growth.max()
    ].symbol.values[0]

    # === Prompt Template ===
    template = """
    Please write a medium-length paragraph that includes some of these information: 
    - The {sector} sub-sector growth in the past one year is {subsec_earning_growth}% for average annual earning growth.
    - The {sector} sub-sector revenue changes in the past one year is {subsec_revenue_growth}% for average annual revenue growth.
    - The {sector} sub-sector places {earn_rank} and {rev_rank} in earnings and revenue rank respectively compared to the other sub-sectors, indicating their performance.
    - From the {sector} sub-sector, {top_earning} is the company with the highest YoY earnings growth and {top_revenue} has the highest YoY revenue growth.
    - Add conditions that uniquely affect the {sector} sector's growth in Indonesia — do not include generic factors applicable to other sub-sectors.
    - Attempt to mention specific regulations that influence the {sector} sector's growth in Indonesia.

    Do not give financial advice. Keep it concise.
    """

    prompt = PromptTemplate.from_template(template)

    # === Chain ===
    chain = prompt | model

    # === Call the model ===
    result = chain.invoke({
        "sector": sector,
        "subsec_earning_growth": round(subsec_earning_growth * 100, 2),
        "subsec_revenue_growth": round(subsec_revenue_growth * 100, 2),
        "earn_rank": earn_rank,
        "rev_rank": rev_rank,
        "top_earning": top_earning,
        "top_revenue": top_revenue,
    })

    result = result.content
    
    try:
        ticker = [top_earning,top_revenue]
        ticker = list(set(ticker))

        for i in ticker:
            result = result.replace(i, f'[#{i}]{df_comrep[df_comrep.symbol == i].company_name.values[0]}[\#{i}]')
    except:
        return result
    
    return result

### Generate description and save it into DB for Sector Growth index description
growthvalue_index = {}

for i in df_sec.sub_sector:
    try:
        growthvalue_index[i] = growth_desc_generator(df_sec,df_comrep,i)
        print(i)
    except:
        print(f"Can't generate {i} growth description")

for sub_sector in growthvalue_index:
    try:
        supabase.table("idx_subsector_metadata").update(
            {"growth_index_description": growthvalue_index[sub_sector]}
        ).eq("sub_sector", sub_sector).execute()
    except:
        logging.error(f"Failed to update description for {sub_sector}.")
