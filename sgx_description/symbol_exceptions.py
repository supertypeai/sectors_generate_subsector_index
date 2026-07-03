# Manually researched SGX data-quality exceptions, current_symbol_treatment maps a symbol
# to how it should be handled when computing sector-level mcap aggregates, positive/negative
# counts, and the notable-companies list. This is a point-in-time snapshot from manual
# research, not a self-maintaining list - see caveats below the dict.

SGX_SYMBOL_TREATMENT = {
    # Dual-currency counters: same underlying share pool quoted twice.
    # Drop the secondary-currency counter entirely, keep only the primary.
    "XWA": {
        "treatment": "exclude",
        "reason": "dual_currency_secondary",
        "primary_symbol": "AWX",
        "company_name": "AEM Holdings",
    },
    "8U7U": {
        "treatment": "exclude",
        "reason": "dual_currency_secondary",
        "primary_symbol": "UD1U",
        "company_name": "IREIT Global",
    },
    "SET": {
        "treatment": "exclude",
        "reason": "dual_currency_secondary",
        "primary_symbol": "SEB",
        "company_name": "Stoneweg Europe Stapled Trust",
    },
    "P7VU": {
        "treatment": "exclude",
        "reason": "dual_currency_secondary",
        # exception to the SGD-default rule: USD counter NS8U is primary here
        "primary_symbol": "NS8U",
        "company_name": "Hutchison Port Holdings Trust",
    },

    "MENU": {
        "treatment": "exclude",
        "reason": "dual_currency_secondary",
        "primary_symbol": "MXNU",
        "company_name": "Elite UK REIT",
    },

    # Phantom scraper entries: no real corresponding SGX listing / maybe right issue / temp.
    "8M4R": {
        "treatment": "exclude",
        "reason": "phantom_scraper_entry",
        "company_name": "Duty Free International (fake duplicate)",
    },
    "YV4R": {
        "treatment": "exclude",
        "reason": "phantom_scraper_entry",
        "company_name": "Duty Free International (fake duplicate)",
    },

    # Rights/warrants/temporary counters: not ordinary equity.
    "LQPR": {
        "treatment": "exclude",
        "reason": "rights_or_warrants_counter",
        "company_name": "AcroMeta Group",
    },
    "VDCR": {
        "treatment": "exclude",
        "reason": "rights_or_warrants_counter",
        "company_name": "AcroMeta Group",
    },
    "Y5DR": {
        "treatment": "exclude",
        "reason": "rights_or_warrants_counter",
        "company_name": "Singapore Institute of Advanced Medicine",
    },

    # Foreign secondary listings: prior trading history exists on another
    # exchange, so a null last_365_mcap here is not a genuine debut. Exclude
    # from aggregate and pos/neg counts, but still eligible for the
    # notable-companies snapshot ranking.
    "AVP": {
        "treatment": "exclude_from_aggregate_only",
        "reason": "foreign_secondary_listing_nasdaq",
        "company_name": "AvePoint Inc",
        "note": "confirm exact SGX ticker before use, recorded as AvePoint / AVP but not independently re-verified here",
    },
    "SEG": {
        "treatment": "exclude_from_aggregate_only",
        "reason": "foreign_secondary_listing_hkex",
        "company_name": "Concord New Energy",
    },

    # Genuinely flat, verified against an external source, not a stale
    # pipeline artifact. Include in aggregate as zero change, exclude from
    # positive/negative counts.
    "5EF": {"treatment": "flat_confirmed", "reason": "verified_genuinely_unchanged"},
    "D8DU": {"treatment": "flat_confirmed", "reason": "verified_genuinely_unchanged"},
    "5GZ": {"treatment": "flat_confirmed", "reason": "verified_genuinely_unchanged"},
    "SGR": {"treatment": "flat_confirmed", "reason": "verified_genuinely_unchanged"},
    "A33": {"treatment": "flat_confirmed", "reason": "verified_genuinely_unchanged"},
    "43E": {"treatment": "flat_confirmed", "reason": "verified_genuinely_unchanged"},

    # Genuine IPOs/spin-offs with no prior trading history anywhere.
    # Fold in as flat using current mcap as baseline, do not exclude.
    "KIN": {"treatment": "flat_confirmed", "reason": "genuine_ipo_fold_in", "company_name": "Kin Global Ltd"},
    "LCC": {"treatment": "flat_confirmed", "reason": "genuine_ipo_fold_in", "company_name": "Lum Chang Creations"},
    "UZF": {"treatment": "flat_confirmed", "reason": "genuine_ipo_fold_in", "company_name": None},
}

# Name correction, not an exclusion. Fix at the source table if possible
# rather than patching at aggregation time.
SGX_NAME_CORRECTIONS = {
    "WNH": "Autagco Ltd",
}