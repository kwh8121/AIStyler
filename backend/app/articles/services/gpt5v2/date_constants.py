#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date formatting constants for Korea Times Style Guide
Based on AP Style with Korea Times modifications
"""

# Month abbreviations (Korea Times Style Guide #10)
# When followed by a specific date (1-31):
#   MUST abbreviate: Jan., Feb., Aug., Sept., Oct., Nov., Dec.
#   NEVER abbreviate: March, April, May, June, July
MONTH_FULL_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

MONTH_ABBREVIATIONS = [
    "", "Jan.", "Feb.", "March", "April", "May", "June",
    "July", "Aug.", "Sept.", "Oct.", "Nov.", "Dec."
]

# Mapping for normalization (handles various input formats)
MONTH_NORMALIZATION = {
    # Full names
    'january': 'January', 'february': 'February', 'march': 'March',
    'april': 'April', 'may': 'May', 'june': 'June', 'july': 'July',
    'august': 'August', 'september': 'September', 'october': 'October',
    'november': 'November', 'december': 'December',

    # Abbreviated forms (with period)
    'jan.': 'Jan.', 'feb.': 'Feb.', 'mar.': 'March', 'apr.': 'April',
    'aug.': 'Aug.', 'sep.': 'Sept.', 'sept.': 'Sept.',
    'oct.': 'Oct.', 'nov.': 'Nov.', 'dec.': 'Dec.',

    # Abbreviated forms (without period)
    'jan': 'Jan.', 'feb': 'Feb.', 'mar': 'March', 'apr': 'April',
    'aug': 'Aug.', 'sep': 'Sept.', 'sept': 'Sept.',
    'oct': 'Oct.', 'nov': 'Nov.', 'dec': 'Dec.',
}

# Day of week names
WEEKDAY_NAMES = [
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
]

# Date context range (Korea Times Style Guide #4)
# Within 7 days: Use days of the week
# Older than 7 days: Use the exact date
DATE_CONTEXT_DAYS = 6  # ±6 days = within 7 days


def get_month_abbreviation(month: str, has_day: bool = True) -> str:
    """
    Get correct month abbreviation per Korea Times Style Guide #10

    Args:
        month: Month name (case-insensitive)
        has_day: Whether a specific day (1-31) follows the month

    Returns:
        Abbreviated month if has_day=True, full name otherwise

    Examples:
        >>> get_month_abbreviation("November", has_day=True)
        'Nov.'
        >>> get_month_abbreviation("November", has_day=False)
        'November'
        >>> get_month_abbreviation("March", has_day=True)
        'March'
    """
    month_lower = month.lower().rstrip('.')

    # Normalize input
    normalized = MONTH_NORMALIZATION.get(month_lower)
    if not normalized:
        return month  # Return as-is if unknown

    # If no day follows, always use full name
    if not has_day:
        # Strip period if present
        return normalized.rstrip('.')

    # If day follows, return abbreviated form
    return normalized


def get_month_number(month: str) -> int:
    """
    Convert month name to number (1-12)

    Args:
        month: Month name (case-insensitive, with or without period)

    Returns:
        Month number (1-12), or 0 if not found

    Examples:
        >>> get_month_number("November")
        11
        >>> get_month_number("Nov.")
        11
    """
    month_lower = month.lower().rstrip('.')

    # Try normalization first
    normalized = MONTH_NORMALIZATION.get(month_lower)
    if normalized:
        clean = normalized.rstrip('.')
        try:
            return MONTH_FULL_NAMES.index(clean)
        except (ValueError, IndexError):
            pass

    # Try full names
    month_cap = month.capitalize()
    try:
        return MONTH_FULL_NAMES.index(month_cap)
    except (ValueError, IndexError):
        return 0
