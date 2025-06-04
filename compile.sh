#!/bin/bash

# Simple compilation script to build the wiki parser

echo "Compiling wiki_parser..."

# Try to use gcc-15 if available, otherwise fall back to gcc
if command -v gcc-15 &> /dev/null; then
    echo "Using gcc-15 for best performance..."
    gcc-15 -O3 -march=native -mtune=native -fomit-frame-pointer -flto -pthread -o wiki_parser wiki_parser.c
else
    echo "Using gcc for compilation..."
    gcc -O3 -march=native -fomit-frame-pointer -flto -pthread -o wiki_parser wiki_parser.c
fi

if [ $? -eq 0 ]; then
    echo "Success! The wiki parser has been created."
    echo ""
    echo "Usage: ./wiki_parser dewiki-20250520-pages-articles-multistream.xml [command]"
    echo "Commands:"
    echo "  count-cats        Count categories in Wikipedia articles (default)"
    echo "  search \"term\" N   Search for a term with N characters of context"
    echo "  extract-isbns     Extract all ISBNs with article titles"
    echo ""
    chmod +x wiki_parser
else
    echo "Failed to compile the parser."
fi
