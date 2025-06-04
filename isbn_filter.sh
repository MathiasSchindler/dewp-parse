#!/bin/bash
# isbn_filter.sh - Filter ISBNs by article name

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <isbn-file> <article-pattern>"
    echo "Filters ISBNs from articles matching the given pattern."
    echo ""
    echo "Examples:"
    echo "  $0 isbns.txt \"Deutschland\"     # Find ISBNs from articles with 'Deutschland' in title"
    echo "  $0 isbns.txt \"^Berlin$\"        # Find ISBNs from the exact article 'Berlin'"
    exit 1
fi

ISBN_FILE="$1"
PATTERN="$2"

if [ ! -f "$ISBN_FILE" ]; then
    echo "Error: File '$ISBN_FILE' not found!"
    exit 1
fi

echo "=== ISBNs from Articles Matching: '$PATTERN' ==="
echo ""

# Count matching articles and ISBNs
MATCHING_ISBNS=$(grep -E "$PATTERN" "$ISBN_FILE")
TOTAL=$(echo "$MATCHING_ISBNS" | wc -l)
ARTICLES=$(echo "$MATCHING_ISBNS" | cut -d'|' -f1 | sort -u | wc -l)

if [ "$TOTAL" -eq 0 ]; then
    echo "No matches found for pattern: '$PATTERN'"
    exit 0
fi

echo "Found $TOTAL ISBNs in $ARTICLES articles matching '$PATTERN'"
echo ""
echo "=== ISBNs from Matching Articles ==="
echo "$MATCHING_ISBNS"
