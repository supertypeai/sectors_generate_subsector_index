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
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

load_dotenv()

# Configure supabase client
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
supabase = create_client(url, key)

# Read Data
def read_table_from_supabase(supabase, table_name):

    response = supabase.table(table_name).select("*").execute()

    response = pd.DataFrame(response.data)

    return response

df_compro = read_table_from_supabase(supabase,"idx_company_profile")
df_comrep = read_table_from_supabase(supabase,"idx_company_report")
df_subsec = read_table_from_supabase(supabase,"idx_subsector_metadata")
df_metrics = read_table_from_supabase(supabase,"idx_calc_metrics_daily")
df_sec = read_table_from_supabase(supabase,"idx_sector_reports")
df_idx = read_table_from_supabase(supabase,"idx_aggregated_calc")

# Data Processing
def get_last_data_from_json(row, last, metrics):
    data_list = ast.literal_eval(str(row))
    last_metrics_data = data_list[last][metrics]
    return last_metrics_data

def sub_sector_financial_processing(df_sec):
    df_sec = df_sec[["sector","slug","sub_sector","weighted_avg_growth_data","weighted_max_drawdown","weighted_rsd_close","historical_valuation"]]

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
    df_metrics = df_metrics[["symbol","price_change_30_days","max_drawdown","rsd_close"]].merge(df_compro[["symbol","company_name",'sub_sector_id']])
    df_metrics = df_metrics.merge(df_subsec[["sub_sector_id","sub_sector","sector"]])

    # Change symbol format
    format_text = lambda company_name, symbol: f"[#{symbol}]{company_name}[\\#{symbol}]"
    df_metrics["symbol_company"] = df_metrics.apply(lambda row: format_text(row['company_name'], row['symbol']), axis=1)

    # Get average changes per sector
    df_changes_sector = df_metrics.groupby(["sub_sector"]).agg({"price_change_30_days":"mean"}).reset_index()
    df_changes_sector.columns = ["sub_sector","sector_price_change_30_days"]

    df_metrics = df_metrics.merge(df_changes_sector, on =["sub_sector"])

    return df_metrics

def company_pe_processing(df_comrep):
    df_comrep = df_comrep[["symbol","company_name","sector","sub_sector","historical_valuation","yoy_quarter_earnings_growth","yoy_quarter_revenue_growth","net_profit_margin"]]

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
    df_no_company = df_compro[["symbol",'sub_sector_id',"listing_date"]].merge(df_subsec[["sub_sector_id","sub_sector","sector"]])

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
    # subsector PE value
    subsec_pe = df_sec[df_sec.sub_sector == sector]["last_pe_value"].values[0]

    # P/E ratio comparison to last year
    if subsec_pe<0:
        status = "negative"
    elif (subsec_pe>0) & (subsec_pe < df_sec[df_sec.sub_sector == sector]["last_two_pe_value"].values[0]):
        status = "lower"
    else:
        status = "higher"

    # PE ranking compared to other subsector
    pe_rank = df_sec[(df_sec.last_pe_value < subsec_pe) & (df_sec.last_pe_value >=0)].shape[0]+1

    # no of company in subsector
    sector_comp_num = df_comrep[(df_comrep.sub_sector == sector)].shape[0]

    # no of company that outperform subsector avg PE
    outperform_sector = df_comrep[(df_comrep.sub_sector == sector) & (df_comrep.pe_value < subsec_pe) & (df_comrep.pe_value > 0)].shape[0]

    # top 3 undervalued company name
    undervalued = df_comrep[(df_comrep.pe_value >=0) & (df_comrep.sub_sector == sector)].sort_values("pe_value", ascending=True).head(3)
    undervalued = undervalued.symbol.values

    # Generate LLM for P/E ratio description
    prompt = """
    Please write a medium-length paragraph that include some of these: 
    - describing whether the {sector} sub-sector is considered defensive or cyclical, along with its historical P/E value conditions categorized as moderate, high, or low. 
    - Explore potential factors influencing the current P/E value {status} compared to last year's P/E value for the {sector} sub-sector, noting that lower P/E values indicate undervaluation and vice versa. 
    - Notably, {outperform_sector} companies from the {sector} boasting favorable P/E ratios below the market average, with {undervalued} standing out as the most undervalued based on P/E ratio analysis. 
    - Provide concrete hypotheses on what said sector command a PE value in that region, and how it relates to the broader Indonesian economy.

    Don't give a financial advice and keep it short, no need to explain in detail for all the factors
    """

    model = ChatOpenAI(temperature=0.5, openai_api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = PromptTemplate(input_variables=["sub_sector","sub_industries","companies"],template=prompt)

    chain = LLMChain(llm=model, prompt=prompt)

    result = chain.run(sector=sector, status=status, pe_rank=pe_rank, outperform_sector=outperform_sector, undervalued=undervalued, sector_comp_num=sector_comp_num)

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
            f"Failed to generate complete description for {i}. The description on the table won't be updated."
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
    # 30 days price changes
    idx_avg_pricechg = df_metrics.price_change_30_days.mean()

    sector_avg_pricechg = df_metrics[df_metrics.sub_sector==sector].sector_price_change_30_days.values[0]
    
    if idx_avg_pricechg>sector_avg_pricechg:
        comparison = "The sub-sector's average price changes is worse than the average IDX price changes in the last 30 days"
    else:
        comparison = "The sub-sector outperform IDX for the average price changes in the last 30 days"

    # New listing company
    new_com_pct = df_nocom[df_nocom.sub_sector == sector].new_company_percentage.values[0]

    if new_com_pct > 0:
        status = f"{new_com_pct}% of the company in {sector} is listing this month"
    else:
        status = f"There aren't any new listing company in this month, the last new listing company in this subsector happened in {np.datetime_as_string(df_nocom[df_nocom.sub_sector == sector].listing_date.values[0], unit='D')}"

    # Price change in last 30 days
    sector_nocomp = df_metrics[(df_metrics.sub_sector == sector)].shape[0]

    pos_price_chg = df_metrics[(df_metrics.sub_sector == sector) & (df_metrics.price_change_30_days > 0)].shape[0]

    neg_price_chg = sector_nocomp - pos_price_chg

    # subsector price changes rank
    df_metrics_subsec = df_metrics.drop_duplicates('sub_sector')
    subsec_rank = df_metrics_subsec[df_metrics_subsec.sector_price_change_30_days > df_metrics_subsec[df_metrics_subsec.sub_sector == sector].sector_price_change_30_days.values[0]].shape[0] + 1

    prompt = """
    Please write a medium-length paragraph that include some of these: 
    - With {sector_nocomp} companies in {sector} sub-sector, this sub-sector placed {subsec_rank} in the sub-sector ranking based on the price changes in the previous 30 days.
    - {comparison}
    - Make a hypoteses based on {comparison}, why that could happened for the {sector} sector
    - In this month, {status}. Add some explanation about barriers to entry based on this month new company status
    - From {sector_nocomp} companies, {pos_price_chg} has a positive price changes in the last 30 days
    - Add the conditions that can affect uniquely to the {sector} market health in Indonesia index but just mention the factor,

    Don't give a financial advice and keep it short, no need to explain in detail for all the factors
    """

    model = ChatOpenAI(temperature=0.5, openai_api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = PromptTemplate(input_variables=["sub_sector","sub_industries","companies"],template=prompt)

    chain = LLMChain(llm=model, prompt=prompt)

    result = chain.run(sector=sector,
    sector_nocomp = sector_nocomp,
    subsec_rank = subsec_rank,
    status = status,
    comparison = comparison,
    pos_price_chg = pos_price_chg)

    return result

### Generate description and save it into DB for Health & Resilience index description
health_index = {}
for i in df_sec.sub_sector:
    try:
        health_index[i] = health_desc_generator(df_metrics,df_nocom,i)
        print(i)
    except:
        logging.error(
            f"Failed to generate complete description for {i}. The description on the table won't be updated."
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
    # Get the earning and revenue growth of the sub_sector
    subsec_earning_growth = df_sec[df_sec.sub_sector == sector].avg_annual_earnings_growth.values[0]
    subsec_revenue_growth = df_sec[df_sec.sub_sector == sector].avg_annual_revenue_growth.values[0]

    # Get the earning and revenue ranking
    earn_rank = df_sec[df_sec.avg_annual_earnings_growth > subsec_earning_growth].shape[0]+1
    rev_rank = df_sec[df_sec.avg_annual_revenue_growth > subsec_revenue_growth].shape[0]+1

    # Get the company for the top earners, top revenue, and to net profit margin
    df_comrep_subsec = df_comrep[(df_comrep.sub_sector == sector)]

    top_earning = df_comrep[(df_comrep.sub_sector == sector) & (df_comrep.yoy_quarter_earnings_growth == df_comrep_subsec.yoy_quarter_earnings_growth.max())].symbol.values[0]
    top_revenue = df_comrep[(df_comrep.sub_sector == sector) & (df_comrep.yoy_quarter_revenue_growth == df_comrep_subsec.yoy_quarter_revenue_growth.max())].symbol.values[0]
    top_netprofmarg = df_comrep[(df_comrep.sub_sector == sector) & (df_comrep.net_profit_margin == df_comrep_subsec.net_profit_margin.max())].symbol.values[0]

    # Specify the prompt
    prompt = """
    Please write a medium-length paragraph that include some of these information: 
    - The {sector} sub-sector growth in the past one year is {subsec_earning_growth}% for average annual earning growth changes
    - The {sector} sub-sector revenue changes in the past one year is {subsec_revenue_growth}% for average annual revenue growth changes.
    - The {sector} sub-sector place {earn_rank} and {rev_rank} for  earnings rank and revenue rank respectively compared to the other sub-sectors. Indicating their performance compare to another sub-sector
    - - From the {sector} sub-sector, {top_earning} is the companies with the highest YoY earning growth, {top_revenue} is the companies with the highest YoY revenue growth, and {top_netprofmarg} is the company with the highest net profit margin.
    - Add the condition that can affect the {sector} sector growth in Indonesia, and use only condition that can affect only to the {sector} sub-sectors don't add any generic answer that can be applied to other sub-sector. 
    - Attempt to provide regulatory that affect the growth unique to {sector} sub-sector in Indonesia

    Don't give a financial advice and keep it short, no need to explain in detail for all the factors
    """
    # Call OpenAI model
    model = ChatOpenAI(temperature=0.5, openai_api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = PromptTemplate(input_variables=["sub_sector","sub_industries","companies"],template=prompt)

    chain = LLMChain(llm=model, prompt=prompt)

    result = chain.run(sector=sector,
    subsec_earning_growth = round(subsec_earning_growth*100,2),
    subsec_revenue_growth = round(subsec_revenue_growth*100,2),
    earn_rank = earn_rank,
    rev_rank = rev_rank,
    top_earning = top_earning,
    top_revenue = top_revenue,
    top_netprofmarg = top_netprofmarg)

    try:
        ticker = [top_earning,top_revenue,top_netprofmarg]
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
