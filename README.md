# Wiki XML Parser

A high-performance C program for parsing large Wikipedia XML dumps using memory mapping (mmap) for maximum speed. Can process **5.8+ million articles in ~32 seconds** (~182,000 articles/second).

## Features

- Memory mapping (mmap) for extremely fast file access
- Multi-threaded processing with optimized thread count
- Boyer-Moore-Horspool algorithm for efficient string searching
- Two-phase parsing approach for scalable multi-threading
- Handles very large article text (100KB+)
- Provides a flexible callback interface for processing articles
- Includes progress reporting and performance statistics

## Building the Parser

```bash
# Simple compilation (automatically uses gcc-15 if available)
./compile.sh

# OR find the optimal thread count for your system and build optimized binary
./benchmark.sh
```

## Usage

```bash
./wiki_parser <wiki-xml-file> [action] [options]
```

### Available Actions:

1. **Count Categories (default)**
   ```bash
   ./wiki_parser dewiki-20250520-pages-articles-multistream.xml count-cats
   ```
   This will analyze all articles and count category usage.

2. **Search for a Term**
   ```bash
   ./wiki_parser dewiki-20250520-pages-articles-multistream.xml search "Deutschland" 100
   ```
   This will find all occurrences of "Deutschland" and save excerpts with 100 characters of context before and after each match to a file named `search-Deutschland.txt`.

3. **Extract ISBNs**
   ```bash
   ./wiki_parser dewiki-20250520-pages-articles-multistream.xml extract-isbns
   ```
   This will extract all ISBNs from all articles and save them to `isbns.txt` in the format `ArticleTitle|ISBN`.
   
   You can also specify a custom output filename:
   ```bash
   ./wiki_parser dewiki-20250520-pages-articles-multistream.xml extract-isbns output.txt
   ```
   
   The parser recognizes ISBNs in multiple MediaWiki formats:
   - Direct mentions: ISBN 978-3-89821-357-8
   - BibISBN templates: {{BibISBN|3-7913-2095-5}}
   - Literatur templates with ISBN parameter: ISBN=0-8131-1803-4

### ISBN Analysis Tools

After extracting ISBNs, you can analyze them using the provided helper scripts:

1. **ISBN Statistics**
   ```bash
   ./isbn_analyzer.sh isbns.txt 15
   ```
   This will analyze the extracted ISBNs and show statistics including:
   - Total ISBNs found and number of articles containing ISBNs
   - Top N articles with the most ISBNs (15 in this example)
   - ISBN format distribution (ISBN-10 vs ISBN-13)
   - Sample ISBNs

2. **ISBN Filtering**
   ```bash
   ./isbn_filter.sh isbns.txt "Deutschland"
   ```
   This will filter and display all ISBNs from articles with "Deutschland" in the title.
   
   You can use regular expressions for more advanced filtering:
   ```bash
   ./isbn_filter.sh isbns.txt "^Berlin$"  # Exact title match
   ```

## Extending the Parser

The unified parser uses a callback-based approach, making it easy to add new functionality:

1. Define a callback function and data structure for your specific need
2. Add a new action in the main function
3. Call `parse_wiki_xml_mt` with your callback

### Example: Creating a Custom Callback

```c
// Data structure for your specific task
typedef struct {
    // Your data fields here
} my_task_data;

// Your callback function
void my_task_callback(const char* title, size_t title_len, 
                      const char* text, size_t text_len,
                      void* user_data) {
    my_task_data* data = (my_task_data*)user_data;
    
    // Process the article title and text
    // ...
}

// In main:
my_task_data data = {0};
parse_wiki_xml_mt(filename, my_task_callback, &data, num_threads);
```

The callback approach allows your function to be called in parallel from multiple threads, so make sure your callback is thread-safe if needed.

## Performance

The unified parser achieves excellent processing speed on the full German Wikipedia dump:
- Single-threaded: ~182,000 articles per second
- Multi-threaded: ~140,000+ articles per second with optimal thread count

## Performance Optimizations

The parser includes all the best performance features:
- Parallel processing using multiple CPU cores with automatic thread count optimization
- Two-phase parsing approach for better multi-threading scalability:
  - Phase 1: Quick scan to identify all page boundaries
  - Phase 2: Distributed processing of pages across all threads
- Boyer-Moore-Horspool algorithm for fast string searching
- Memory mapping for efficient file access
- CPU-specific optimizations via compiler flags
- GCC 15 compiler support for additional performance

To compile the optimized version:
```bash
./compile.sh
```

### Finding Optimal Thread Count

For peak performance, find the optimal thread count for your specific hardware:

```bash
# Run the benchmark script to find the ideal thread count and create an optimized binary
./benchmark.sh
```

This script will:
1. Create a 1GB sample file (if it doesn't exist) for faster benchmarking
2. Test different thread counts (1-32, limited by your CPU core count)
3. Create an optimized executable (`wiki_parser`) with the best thread count hard-coded

The benchmark script runs 3 tests per thread count to find the most consistent performance.

### How Thread Optimization Works

The unified parser uses a smart approach to thread management:
1. It defines a `OPTIMAL_THREAD_COUNT` constant that can be set at compile time
2. If this constant is set (non-zero), it uses that thread count
3. If not set, it detects your system's CPU core count
4. The benchmark script tests different thread counts and compiles with the best one

### Recommended Usage

For the best performance:

1. First run the benchmark to create an optimized binary:
```bash
./benchmark.sh
```

2. Then use the optimized version for all operations:
```bash
# Count categories (using optimal thread count)
./wiki_parser dewiki-20250520-pages-articles-multistream.xml count-cats

# Search for a term with context (using optimal thread count)
./wiki_parser dewiki-20250520-pages-articles-multistream.xml search "Deutschland" 100
```

The optimized version automatically uses the best thread count determined by benchmarking, without requiring you to specify thread count each time.
