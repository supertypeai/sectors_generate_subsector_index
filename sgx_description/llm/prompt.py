from pydantic import Field, BaseModel


class HealthDescription(BaseModel):
    """Structured output for a sector's 30-day price momentum/health narrative.

    Produced from `user_prompt_health()`, grounded in each sector's median
    price change, its rank against other sectors, and the share of
    companies with positive vs negative 30-day returns.
    """

    description: str = Field(
        description=(
            "A medium-length, concise paragraph describing the sector's price "
            "momentum/health over the last 30 days. Must be grounded strictly in "
            "the sector metrics provided (median price change, sector rank, "
            "positive/negative company counts) - never invent figures. Should "
            "include a hypothesis for why the sector is behaving this way, and "
            "any Singapore-market-specific factors currently affecting it. Do not "
            "state a specific calendar year or exact date, and do not give "
            "financial advice."
        )
    )

    reasoning: str = Field(
        description=(
            "Brief explanation of which input facts were used and why, and how "
            "they were combined to arrive at the description above - not shown "
            "to end users, used only for debugging/auditing the generation."
        )
    )


class PEDescription(BaseModel):
    """Structured output for a sector's P/E valuation narrative.

    Produced from `user_prompt_pe()`, grounded in the sector's own P/E
    median, the broader SGX-wide P/E median, the year-over-year trend, and
    a short list of the sector's most undervalued companies by P/E.
    """

    description: str = Field(
        description=(
            "A medium-length, concise paragraph describing the sector's P/E "
            "valuation. Must be grounded strictly in the provided metrics "
            "(sector P/E median, SGX-wide median, trend vs last year, "
            "undervalued company examples) - never invent or restate the exact "
            "numeric P/E values themselves, describe direction/magnitude "
            "qualitatively instead. Reference undervalued companies by full "
            "company name only, never by ticker symbol. Do not state a specific "
            "calendar year or exact date, and do not give financial advice."
        )
    )

    reasoning: str = Field(
        description=(
            "Brief explanation of which input facts were used and why, and how "
            "they were combined to arrive at the description above - not shown "
            "to end users, used only for debugging/auditing the generation."
        )
    )


class GrowthDescription(BaseModel):
    """Structured output for a sector's earnings/revenue growth narrative.

    Produced from `user_prompt_growth()`, grounded in the sector's average
    annual earnings/revenue growth, its rank against other sectors for
    each metric, and the company with the highest yearly earnings growth
    and the company with the highest yearly revenue growth in the sector.
    """

    description: str = Field(
        description=(
            "A medium-length, concise paragraph describing the sector's earnings "
            "and revenue growth over the past year. Must be grounded strictly in "
            "the provided metrics (average annual earnings/revenue growth, "
            "earnings rank, revenue rank, top earnings/revenue growth companies) - "
            "never invent figures, and never conflate the earnings rank with the "
            "revenue rank. Reference the top-growth companies by full company "
            "name only, never by ticker symbol. Should include Singapore-market "
            "or sector-specific factors that uniquely affect this sector's "
            "growth. Do not state a specific calendar year or exact date, and do "
            "not give financial advice."
        )
    )

    reasoning: str = Field(
        description=(
            "Brief explanation of which input facts were used and why, and how "
            "they were combined to arrive at the description above - not shown "
            "to end users, used only for debugging/auditing the generation."
        )
    )


class PromptCollections: 
    @staticmethod
    def system_prompt():
        return """ 
            You are a financial content writer for Sectors, a data platform covering
            the Singapore Exchange (SGX). You generate short analytical description
            paragraphs for SGX sectors based strictly on the metrics and context
            provided to you.
        """

    @staticmethod
    def user_prompt_health():
        return """ 
            Please write a medium-length paragraph that includes some of the
            following facts. The order below is not the required sentence order;
            choose whichever fact makes the strongest natural opening sentence.

            - With {total_companies} companies in the {sector} sector, this sector
            placed {sector_rank} ut of {total_sectors}  in the sector ranking 
            based on the median price changes in the previous 30 days.

            - Make a hypothesis or explanation based on {comparison_signal}, why that could happen 
            for the {sector} sector with the recent information {year}.
            
            - From {total_companies} companies, {positive_price_change_count} has
            a positive price change in the last 30 days.

            - Add conditions that can uniquely affect the {sector} market health in
            the Singapore index, research for current year {year} — just mention the factors, 
            no need to elaborate.

            - Never state a specific calendar year or exact date in the output
            - Do not introduce an explanation by labeling it as a hypothesis or by using
            hedge phrases such as "a plausible hypothesis"
        
            Ensure return in the following json: 
            {format_instructions}
        """

    @staticmethod
    def user_prompt_pe():
        return """ 
            Write a medium-length paragraph about the {sector} sector's P/E valuation,
            using some of the following facts. The order below is not the required sentence order;
            choose whichever fact makes the strongest natural opening sentence.

            - Based on general knowledge of the sector, describe whether {sector} is
            typically considered defensive or cyclical, and whether a P/E of
            {sector_pe_median} is moderate, high, or low for that kind of sector.
            - The sector's current P/E median is {sector_pe_median}, which is
            {trend_description}. Explore what could explain this.
            - Out of {company_count} companies in the sector, {below_sector_median_count}
            have a P/E below the sector's OWN median. Separately, {below_market_median_count}
            companies have a P/E below the broader SGX-wide median of {overall_pe_median}.
            These are two different comparisons - do not merge them into one figure or
            describe one as if it were the other.
            - Among companies with the lowest positive P/E in the sector, {undervalued_companies}
            are worth naming as examples.
            - Hypothesize what might be driving this sector's P/E levels and how it relates
            to the broader Singapore economy.

            Rules:
            - Write in complete, natural sentences throughout. Do not use em dashes or
            interrupted clauses to list or contrast numbers (avoid patterns like
            "84 firms - eight companies -"). Every figure should read as part of a
            full sentence, not a juxtaposed fragment.
            - When mentioning any company from the undervalued companies list, refer to it
            by its exact full company name only. Never write its ticker symbol, and
            never put the symbol in parentheses next to the name - the symbol is
            provided only so you can identify the company, not to be repeated in your
            writing.
            - The P/E figures above (sector median, market median, prior-year figure) are
            for your own reasoning only. Do NOT state any of these exact P/E values in
            your response. Describe direction and magnitude qualitatively instead
            - Never use the unqualified phrase "market average" - always specify whether a
            figure refers to the sector's own median or the SGX-wide median.
            - Avoid giving financial advice and keep it concise.

            Ensure return in the following json:
            {format_instructions}
        """

    @staticmethod
    def user_prompt_growth():
        return """
            Write a medium-length paragraph about the {sector} sector's earnings and
            revenue growth, using some of the following facts. The order below is not
            the required sentence order; choose whichever fact makes the strongest
            natural opening sentence.

            - The {sector} sector's average annual earnings growth over the past year
            is {avg_annual_earning_growth}%, and its average annual revenue growth is
            {avg_annual_revenue_growth}%.
            - Out of {total_sectors} sectors, {sector} places {earnings_rank} for
            earnings growth and {revenue_rank} for revenue growth. These are two
            different rankings - do not merge them into one figure or describe one
            as if it were the other.
            - {top_earning} is the company with the highest yearly earnings growth in
            the sector, and {top_revenue} has the highest yearly revenue growth.
            - Add conditions that uniquely affect the {sector} sector's growth in
            Singapore, do not include generic factors applicable to other sectors.

            Rules:
            - Write in complete, natural sentences throughout. Do not use em dashes or
            interrupted clauses to list or contrast numbers.
            - When mentioning {top_earning} or {top_revenue}, refer to each by its exact
            full company name only. Never write its ticker symbol.
            - Never state a specific calendar year or exact date in the output.
            - Avoid giving financial advice and keep it concise.

            Ensure return in the following json:
            {format_instructions}
        """