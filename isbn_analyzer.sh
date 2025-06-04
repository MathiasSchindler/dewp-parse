#!/bin/bash
# isbn_stats.sh - Analyze extracted ISBNs

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <isbn-file> [top-n]"
    echo "Analyzes ISBN data extracted from Wikipedia."
    echo ""
    echo "Example:"
    echo "  $0 isbns.txt 20"
    exit 1
fi

ISBN_FILE="$1"
TOP_N="${2:-10}"  # Default to top 10 if not specified

if [ ! -f "$ISBN_FILE" ]; then
    echo "Error: File '$ISBN_FILE' not found!"
    exit 1
fi

echo "=== ISBN Statistics ==="
echo ""

# Total ISBNs
TOTAL=$(wc -l < "$ISBN_FILE")
echo "Total ISBNs found: $TOTAL"

# Articles with ISBNs
ARTICLES=$(cut -d'|' -f1 < "$ISBN_FILE" | sort -u | wc -l)
echo "Articles containing ISBNs: $ARTICLES"

# Average ISBNs per article
if [ "$ARTICLES" -gt 0 ]; then
    AVERAGE=$(echo "scale=2; $TOTAL / $ARTICLES" | bc)
    echo "Average ISBNs per article: $AVERAGE"
fi

echo ""
echo "=== Top $TOP_N Articles by ISBN Count ==="

# List top articles by ISBN count
cut -d'|' -f1 < "$ISBN_FILE" | sort | uniq -c | sort -nr | head -n "$TOP_N"

echo ""
echo "=== ISBN Format Distribution ==="

echo ""
echo "NOTE: Format statistics are approximate due to ISBN variations."
echo "For accurate format analysis, a more sophisticated parser is needed."

# Basic approximation of ISBN-10 vs ISBN-13
# ISBN-13 typically starts with 978- or 979- and has 13 digits
ISBN13=$(grep -E '\|97[89][-0-9]{10,14}$' "$ISBN_FILE" | wc -l)
ISBN10=$(grep -v -E '\|97[89][-0-9]{10,14}$' "$ISBN_FILE" | wc -l)

echo "ISBN-13 format (starts with 978/979): $ISBN13 ($(echo "scale=2; $ISBN13*100/$TOTAL" | bc)%)"
echo "Other formats (likely ISBN-10): $ISBN10 ($(echo "scale=2; $ISBN10*100/$TOTAL" | bc)%)"

echo ""
echo "=== Sample ISBNs ==="
head -n 10 "$ISBN_FILE"
