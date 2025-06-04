#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <ctype.h>
#include <pthread.h>

// Forward declaration for buffer cleanup - removed to avoid type conflicts

/* Thread count optimization */
#ifndef OPTIMAL_THREAD_COUNT
#define OPTIMAL_THREAD_COUNT 0  // Will be determined at runtime if not specified
#endif

// Define callback function type for processing articles
typedef void (*article_callback)(const char* title, size_t title_len, 
                               const char* text, size_t text_len, 
                               void* user_data);

// Boyer-Moore-Horspool algorithm for faster string search
void prepare_bad_char_table(const char* pattern, size_t pattern_len, size_t bad_char[256]) {
    for (size_t i = 0; i < 256; i++) {
        bad_char[i] = pattern_len;
    }
    
    for (size_t i = 0; i < pattern_len - 1; i++) {
        bad_char[(unsigned char)pattern[i]] = pattern_len - 1 - i;
    }
}

// Fast string search using Boyer-Moore-Horspool algorithm
char* fast_strstr(const char* haystack, const char* needle, size_t haystack_len, size_t needle_len) {
    if (needle_len > haystack_len || needle_len == 0) return NULL;
    
    size_t bad_char[256];
    prepare_bad_char_table(needle, needle_len, bad_char);
    
    size_t i = needle_len - 1;
    while (i < haystack_len) {
        size_t j = needle_len - 1;
        while (haystack[i] == needle[j]) {
            if (j == 0) return (char*)(haystack + i);
            i--;
            j--;
        }
        i += bad_char[(unsigned char)haystack[i]] > needle_len - j ? 
              bad_char[(unsigned char)haystack[i]] : needle_len - j;
    }
    
    return NULL;
}

// Structure to hold information about a single page/article
typedef struct {
    size_t start_pos;  // Position in the file
    size_t length;     // Length of the page
} page_info_t;

// Structure to hold work queue for threads
typedef struct {
    char* file_content;
    size_t file_size;
    article_callback callback;
    void* user_data;
    page_info_t* pages;
    size_t total_pages;
    size_t pages_per_thread;
    int num_threads;
    int thread_id;
    size_t articles_processed;
    double start_time;
} thread_data_t;

// Find the position of the title within a page
void extract_title_and_text(const char* page_start, size_t page_len,
                         char** title_start, size_t* title_len,
                         char** text_start, size_t* text_len) {
    
    // Constants for XML tags
    const char* title_start_tag = "<title>";
    const size_t title_start_len = strlen(title_start_tag);
    const char* title_end_tag = "</title>";
    const size_t title_end_len = strlen(title_end_tag);
    const char* text_start_tag = "<text";
    const size_t text_start_len = strlen(text_start_tag);
    const char* text_end_tag = "</text>";
    const size_t text_end_len = strlen(text_end_tag);
    
    // Initialize to not found
    *title_start = NULL;
    *title_len = 0;
    *text_start = NULL;
    *text_len = 0;
    
    // Find the title
    char* temp_title_start = fast_strstr(page_start, title_start_tag, page_len, title_start_len);
    if (!temp_title_start) return;  // No title found
    
    temp_title_start += title_start_len;  // Skip the opening tag
    size_t remaining = page_start + page_len - temp_title_start;
    char* title_end = fast_strstr(temp_title_start, title_end_tag, remaining, title_end_len);
    if (!title_end) return;  // Malformed XML
    
    *title_start = temp_title_start;
    *title_len = title_end - temp_title_start;
    
    // Find the text
    char* temp_text_start = fast_strstr(title_end, text_start_tag, page_start + page_len - title_end, text_start_len);
    if (!temp_text_start) return;  // No text found
    
    // Find the closing > of the text tag
    char* text_content_start = strchr(temp_text_start, '>');
    if (!text_content_start) return;  // Malformed XML
    text_content_start++;  // Skip the >
    
    remaining = page_start + page_len - text_content_start;
    char* text_end = fast_strstr(text_content_start, text_end_tag, remaining, text_end_len);
    if (!text_end) return;  // Malformed XML
    
    *text_start = text_content_start;
    *text_len = text_end - text_content_start;
}

// Worker thread function for processing pages
void* process_pages_thread(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    
    // Calculate the range of pages this thread will process
    size_t start_idx = data->thread_id * data->pages_per_thread;
    size_t end_idx = (data->thread_id == data->num_threads - 1) ? 
                      data->total_pages : 
                      start_idx + data->pages_per_thread;
    
    printf("Thread %d: pages %zu to %zu\n", data->thread_id, start_idx, end_idx - 1);
    
    // Process each page assigned to this thread
    size_t processed = 0;
    double report_threshold = 10000; // Report every 10k articles
    
    for (size_t i = start_idx; i < end_idx; i++) {
        page_info_t page = data->pages[i];
        char* page_start = data->file_content + page.start_pos;
        
        char* title_start;
        size_t title_len;
        char* text_start;
        size_t text_len;
        
        extract_title_and_text(page_start, page.length,
                              &title_start, &title_len,
                              &text_start, &text_len);
        
        if (title_start && text_start) {
            data->callback(title_start, title_len, text_start, text_len, data->user_data);
            processed++;
            
            // Periodically report progress
            if (processed >= report_threshold) {
                double elapsed = (double)(time(NULL) - data->start_time);
                double rate = (elapsed > 0) ? processed / elapsed : 0;
                double percent = (double)processed / (end_idx - start_idx) * 100.0;
                printf("Thread %d: Processed %zu articles (%.2f%%), %.1f articles/sec\n", 
                       data->thread_id, processed, percent, rate);
                report_threshold += 10000; // Report every additional 10k
            }
        }
    }
    
    data->articles_processed = processed;
    return NULL;
}

// First pass: scan through the file to find all page boundaries
size_t find_all_pages(char* file_content, size_t file_size, page_info_t** pages_out) {
    const char* page_start_tag = "<page>";
    const size_t page_start_len = strlen(page_start_tag);
    const char* page_end_tag = "</page>";
    const size_t page_end_len = strlen(page_end_tag);
    
    size_t max_pages = 1000000;  // Initial capacity
    size_t found_pages = 0;
    page_info_t* pages = malloc(max_pages * sizeof(page_info_t));
    if (!pages) {
        perror("Failed to allocate memory for pages");
        return 0;
    }
    
    printf("First pass: locating all pages...\n");
    
    char* current = file_content;
    char* end = file_content + file_size;
    size_t report_threshold = 1000000; // Report progress every million pages
    clock_t start_time = clock();
    
    while (current < end) {
        size_t remaining = end - current;
        
        // Find start of next page
        char* page_start = fast_strstr(current, page_start_tag, remaining, page_start_len);
        if (!page_start) break; // No more pages
        
        remaining = end - page_start;
        char* page_end = fast_strstr(page_start, page_end_tag, remaining, page_end_len);
        if (!page_end) break; // Malformed XML
        
        page_end += page_end_len; // Include the closing tag
        
        // Add this page to our array
        if (found_pages >= max_pages) {
            max_pages *= 2;
            page_info_t* new_pages = realloc(pages, max_pages * sizeof(page_info_t));
            if (!new_pages) {
                perror("Failed to reallocate memory for pages");
                free(pages);
                return 0;
            }
            pages = new_pages;
        }
        
        pages[found_pages].start_pos = page_start - file_content;
        pages[found_pages].length = page_end - page_start;
        found_pages++;
        
        if (found_pages >= report_threshold) {
            double percentage = (page_end - file_content) * 100.0 / file_size;
            printf("Found %zu pages (%.1f%%)\n", found_pages, percentage);
            report_threshold += 1000000;
        }
        
        current = page_end;
    }
    
    // Add any remaining articles to the counter
    if (found_pages > 0) {
        double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
        printf("Found %zu pages\n", found_pages);
        printf("Page scan completed in %.1f seconds\n", elapsed);
    }
    
    *pages_out = pages;
    return found_pages;
}

// Multi-threaded Wikipedia XML parser with two-phase approach
int parse_wiki_xml_mt(const char* filename, article_callback callback, void* user_data, int num_threads) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("Failed to open file");
        return 1;
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Failed to get file size");
        close(fd);
        return 1;
    }
    
    printf("File size: %.2f GB\n", (double)sb.st_size / (1024*1024*1024));
    
    char* file_content = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_content == MAP_FAILED) {
        perror("Failed to memory map the file");
        close(fd);
        return 1;
    }
    
    // Use the pre-determined optimal thread count if available
    if (OPTIMAL_THREAD_COUNT > 0) {
        num_threads = OPTIMAL_THREAD_COUNT;
    } else {
        // Determine number of threads to use based on available cores if not specified
        if (num_threads <= 0) {
            num_threads = sysconf(_SC_NPROCESSORS_ONLN);
            if (num_threads <= 0) num_threads = 4;  // Default if we can't detect
        }
    }
    
    printf("Using %d threads\n", num_threads);
    
    // Phase 1: Find all pages in the file
    page_info_t* pages = NULL;
    size_t total_pages = find_all_pages(file_content, sb.st_size, &pages);
    
    if (total_pages == 0) {
        munmap(file_content, sb.st_size);
        close(fd);
        return 1;
    }
    
    // Phase 2: Process the pages in parallel
    size_t pages_per_thread = total_pages / num_threads;
    if (pages_per_thread == 0) pages_per_thread = 1;
    
    printf("Assigning approximately %zu pages per thread\n", pages_per_thread);
    
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    thread_data_t* thread_data = malloc(num_threads * sizeof(thread_data_t));
    
    time_t start_time = time(NULL);
    
    // Initialize and start worker threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].file_content = file_content;
        thread_data[i].file_size = sb.st_size;
        thread_data[i].callback = callback;
        thread_data[i].user_data = user_data;
        thread_data[i].pages = pages;
        thread_data[i].total_pages = total_pages;
        thread_data[i].pages_per_thread = pages_per_thread;
        thread_data[i].num_threads = num_threads;
        thread_data[i].thread_id = i;
        thread_data[i].articles_processed = 0;
        thread_data[i].start_time = start_time;
        
        if (pthread_create(&threads[i], NULL, process_pages_thread, &thread_data[i]) != 0) {
            perror("Failed to create thread");
            free(threads);
            free(thread_data);
            free(pages);
            munmap(file_content, sb.st_size);
            close(fd);
            return 1;
        }
    }
    
    // Wait for all threads to finish
    size_t total_articles_processed = 0;
    for (int i = 0; i < num_threads; i++) {
        if (pthread_join(threads[i], NULL) != 0) {
            perror("Failed to join thread");
        }
        printf("Thread %d complete: processed %zu articles\n", i, thread_data[i].articles_processed);
        total_articles_processed += thread_data[i].articles_processed;
    }
    
    // Calculate overall performance
    double elapsed = difftime(time(NULL), start_time);
    elapsed = (elapsed == 0) ? 0.1 : elapsed; // Avoid division by zero
    double rate = total_articles_processed / elapsed;
    
    printf("Processing complete. Processed %zu articles in %.1f seconds (%.1f articles/sec)\n", 
           total_articles_processed, elapsed, rate);
    
    // Clean up
    free(threads);
    free(thread_data);
    free(pages);
    munmap(file_content, sb.st_size);
    close(fd);
    
    return 0;
}

// ======== SPECIFIC ARTICLE PROCESSING FUNCTIONS ========

// Structure for counting categories
typedef struct {
    size_t total_articles;
    size_t articles_with_cats;
    size_t total_categories;
    size_t max_categories;
    char max_cat_article[256];
} category_stats_t;

// Process a single article for category counting
void count_categories_callback(const char* title, size_t title_len, 
                             const char* text, size_t text_len, 
                             void* user_data) {
    category_stats_t* stats = (category_stats_t*)user_data;
    stats->total_articles++;
    
    // Find all category links
    const char* category_tag = "[[Kategorie:";
    const size_t category_tag_len = strlen(category_tag);
    
    size_t categories = 0;
    const char* pos = text;
    const char* text_end = text + text_len;
    
    while (pos < text_end) {
        size_t remaining = text_end - pos;
        pos = fast_strstr(pos, category_tag, remaining, category_tag_len);
        if (!pos) break;
        
        categories++;
        pos += category_tag_len;
    }
    
    if (categories > 0) {
        stats->articles_with_cats++;
        stats->total_categories += categories;
        
        if (categories > stats->max_categories) {
            stats->max_categories = categories;
            
            // Copy the article title (safely)
            size_t copy_len = title_len < 255 ? title_len : 255;
            memcpy(stats->max_cat_article, title, copy_len);
            stats->max_cat_article[copy_len] = '\0';
        }
    }
}

// Search for a term in articles and save context
typedef struct {
    const char* search_term;
    size_t search_term_len;
    int context_size;
    FILE* output_file;
} search_context_t;

// Process a single article for searching
void search_text_callback(const char* title, size_t title_len, 
                         const char* text, size_t text_len, 
                         void* user_data) {
    search_context_t* ctx = (search_context_t*)user_data;
    
    // Find all occurrences of the search term
    const char* pos = text;
    const char* text_end = text + text_len;
    
    while (pos < text_end) {
        size_t remaining = text_end - pos;
        const char* match = fast_strstr(pos, ctx->search_term, remaining, ctx->search_term_len);
        if (!match) break;
        
        // Write the match to the output file
        fprintf(ctx->output_file, "Article: %.*s\n", (int)title_len, title);
        
        // Determine context boundaries
        const char* context_start = match - ctx->context_size;
        if (context_start < text) context_start = text;
        
        const char* context_end = match + ctx->search_term_len + ctx->context_size;
        if (context_end > text_end) context_end = text_end;
        
        fprintf(ctx->output_file, "...%.*s...\n\n", 
               (int)(context_end - context_start), context_start);
        
        pos = match + 1;  // Move past the current match to find the next one
    }
}

// Structure for thread-local ISBN buffering
typedef struct {
    char** isbn_buffer;     // Buffer to store ISBNs before writing
    int buffer_size;        // Size of each buffer entry
    int buffer_count;       // Number of entries in buffer
    int buffer_capacity;    // Maximum buffer entries
} isbn_buffer_t;

// Structure for ISBN extraction
typedef struct {
    FILE* output_file;
    pthread_mutex_t mutex;  // Protect the output file
    int found_isbns;        // Count of found ISBNs
    time_t start_time;      // Timestamp for performance calculation
    int buffer_size;        // Size of thread-local buffers
    int buffer_capacity;    // Maximum items per buffer before flushing
} isbn_context_t;

// Check if a character is valid in an ISBN (digit, hyphen, X for checksum)
int is_valid_isbn_char(char c) {
    return isdigit(c) || c == '-' || c == 'X' || c == 'x';
}

// Initialize a thread-local ISBN buffer
isbn_buffer_t* init_isbn_buffer(int buffer_capacity, int buffer_size) {
    isbn_buffer_t* buffer = malloc(sizeof(isbn_buffer_t));
    if (!buffer) return NULL;
    
    buffer->isbn_buffer = malloc(buffer_capacity * sizeof(char*));
    if (!buffer->isbn_buffer) {
        free(buffer);
        return NULL;
    }
    
    for (int i = 0; i < buffer_capacity; i++) {
        buffer->isbn_buffer[i] = malloc(buffer_size);
        if (!buffer->isbn_buffer[i]) {
            // Clean up already allocated buffers
            for (int j = 0; j < i; j++) {
                free(buffer->isbn_buffer[j]);
            }
            free(buffer->isbn_buffer);
            free(buffer);
            return NULL;
        }
    }
    
    buffer->buffer_size = buffer_size;
    buffer->buffer_count = 0;
    buffer->buffer_capacity = buffer_capacity;
    
    return buffer;
}

// Free a thread-local ISBN buffer
void free_isbn_buffer(isbn_buffer_t* buffer) {
    if (!buffer) return;
    
    if (buffer->isbn_buffer) {
        for (int i = 0; i < buffer->buffer_capacity; i++) {
            if (buffer->isbn_buffer[i]) {
                free(buffer->isbn_buffer[i]);
            }
        }
        free(buffer->isbn_buffer);
    }
    
    free(buffer);
}

// Flush the buffer to the output file
void flush_isbn_buffer(isbn_buffer_t* buffer, isbn_context_t* ctx) {
    if (!buffer || buffer->buffer_count == 0) return;
    
    pthread_mutex_lock(&ctx->mutex);
    
    // Write all buffered ISBNs to file
    for (int i = 0; i < buffer->buffer_count; i++) {
        fputs(buffer->isbn_buffer[i], ctx->output_file);
    }
    
    // Update the counter
    ctx->found_isbns += buffer->buffer_count;
    
    // Report progress
    if ((ctx->found_isbns / 10000) > ((ctx->found_isbns - buffer->buffer_count) / 10000)) {
        printf("Found %d ISBNs so far\n", ctx->found_isbns);
    }
    
    pthread_mutex_unlock(&ctx->mutex);
    
    // Reset the buffer count
    buffer->buffer_count = 0;
}

// Add an ISBN to the buffer
void add_to_buffer(isbn_buffer_t* buffer, const char* title, size_t title_len, 
                  const char* isbn, size_t isbn_len, isbn_context_t* ctx) {
    
    // If buffer is full, flush it
    if (buffer->buffer_count >= buffer->buffer_capacity) {
        flush_isbn_buffer(buffer, ctx);
    }
    
    // Format the ISBN entry
    snprintf(buffer->isbn_buffer[buffer->buffer_count], buffer->buffer_size,
            "%.*s|%.*s\n", (int)title_len, title, (int)isbn_len, isbn);
    
    buffer->buffer_count++;
}

// Validate if a string could be a valid ISBN
// Basic validation: at least 10 digits, not too long, has valid characters
int is_valid_isbn_format(const char* isbn_start, const char* isbn_end) {
    size_t total_len = isbn_end - isbn_start;
    if (total_len < 9 || total_len > 17) {
        return 0; // Too short or too long for an ISBN
    }
    
    // Count digits
    int digit_count = 0;
    for (const char* p = isbn_start; p < isbn_end; p++) {
        if (isdigit(*p)) digit_count++;
    }
    
    // ISBN-10 has 10 digits, ISBN-13 has 13 digits
    return (digit_count == 10 || digit_count == 13);
}

// Optimized process for extracting ISBNs from article text
void extract_isbns_callback(const char* title, size_t title_len, 
                           const char* text, size_t text_len, 
                           void* user_data) {
    isbn_context_t* ctx = (isbn_context_t*)user_data;
    
    // Create thread-local buffer if it doesn't exist in thread-local storage
    static __thread isbn_buffer_t* buffer = NULL;
    if (!buffer) {
        buffer = init_isbn_buffer(ctx->buffer_capacity, ctx->buffer_size);
        if (!buffer) {
            // Fall back to direct writes if buffer allocation fails
            fprintf(stderr, "Warning: Failed to allocate ISBN buffer\n");
        }
    }
    
    // Patterns to search for (for efficient search)
    const char* isbn_markers[] = {"ISBN", "{{BibISBN|", "ISBN="};
    const size_t isbn_markers_len[] = {4, 10, 5};
    const int num_markers = 3;
    
    // Scan through the text
    const char* pos = text;
    const char* text_end = text + text_len;
    
    // Use a maximum skip table approach (find closest potential marker)
    while (pos < text_end) {
        // Find the closest marker
        const char* closest_match = NULL;
        int closest_marker = -1;
        
        for (int i = 0; i < num_markers; i++) {
            size_t remaining = text_end - pos;
            const char* match = fast_strstr(pos, isbn_markers[i], remaining, isbn_markers_len[i]);
            
            if (match && (!closest_match || match < closest_match)) {
                closest_match = match;
                closest_marker = i;
            }
        }
        
        // If no marker found, we're done
        if (!closest_match) {
            break;
        }
        
        // Process the marker based on its type
        const char* isbn_start = NULL;
        const char* isbn_end = NULL;
        
        if (closest_marker == 0) {  // Simple ISBN
            isbn_start = closest_match + isbn_markers_len[0];
            
            // Skip whitespace
            while (isbn_start < text_end && (*isbn_start == ' ' || *isbn_start == ':' || *isbn_start == '=')) {
                isbn_start++;
            }
            
            // Find end of ISBN
            isbn_end = isbn_start;
            while (isbn_end < text_end && (is_valid_isbn_char(*isbn_end) || *isbn_end == ' ')) {
                isbn_end++;
            }
            
            // Trim trailing spaces/hyphens
            while (isbn_end > isbn_start && (*(isbn_end-1) == ' ' || *(isbn_end-1) == '-')) {
                isbn_end--;
            }
        }
        else if (closest_marker == 1) {  // BibISBN template
            isbn_start = closest_match + isbn_markers_len[1];
            isbn_end = isbn_start;
            
            // Find end of parameter
            while (isbn_end < text_end && *isbn_end != '|' && *isbn_end != '}') {
                isbn_end++;
            }
        }
        else {  // Literatur template ISBN parameter
            isbn_start = closest_match + isbn_markers_len[2];
            
            // Skip whitespace/equals
            while (isbn_start < text_end && (*isbn_start == ' ' || *isbn_start == '=')) {
                isbn_start++;
            }
            
            // Find end of parameter
            isbn_end = isbn_start;
            while (isbn_end < text_end && *isbn_end != '|' && *isbn_end != '}') {
                isbn_end++;
            }
        }
        
        // Check if ISBN is valid
        if (isbn_start && isbn_end > isbn_start && is_valid_isbn_format(isbn_start, isbn_end)) {
            if (buffer) {
                add_to_buffer(buffer, title, title_len, isbn_start, isbn_end - isbn_start, ctx);
            }
            else {
                // Direct write if buffer allocation failed
                pthread_mutex_lock(&ctx->mutex);
                fprintf(ctx->output_file, "%.*s|%.*s\n", 
                       (int)title_len, title, (int)(isbn_end - isbn_start), isbn_start);
                ctx->found_isbns++;
                pthread_mutex_unlock(&ctx->mutex);
            }
        }
        
        // Move past this match (skip ahead more than 1 character)
        pos = closest_match + 1;
        
        // Optimize search by skipping ahead if we can
        if (isbn_end > pos) {
            pos = isbn_end;
        }
    }
    
    // Flush buffer at the end of the article
    if (buffer) {
        flush_isbn_buffer(buffer, ctx);
    }
}

// ======== MAIN FUNCTION ========

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <wiki-xml-file> [action] [options]\n", argv[0]);
        printf("Actions:\n");
        printf("  count-cats         Count categories in articles (default)\n");
        printf("  search \"term\" N    Search for a term with N characters of context\n");
        printf("  extract-isbns      Extract all ISBNs with article titles\n");
        return 1;
    }
    
    const char* filename = argv[1];
    const char* action = (argc >= 3) ? argv[2] : "count-cats";
    int num_threads = (argc >= 4) ? atoi(argv[3]) : 0;
    
    if (strcmp(action, "count-cats") == 0) {
        printf("Counting categories in Wikipedia articles...\n");
        
        category_stats_t stats = {0};
        parse_wiki_xml_mt(filename, count_categories_callback, &stats, num_threads);
        
        printf("\nCategory Statistics:\n");
        printf("Total articles: %zu\n", stats.total_articles);
        printf("Articles with categories: %zu (%.1f%%)\n", 
              stats.articles_with_cats,
              (stats.total_articles > 0) ? 
                  (double)stats.articles_with_cats / stats.total_articles * 100 : 0.0);
        printf("Total categories: %zu\n", stats.total_categories);
        printf("Average categories per article: %.2f\n",
              (stats.articles_with_cats > 0) ?
                  (double)stats.total_categories / stats.articles_with_cats : 0.0);
        printf("Maximum categories: %zu in article \"%s\"\n", 
              stats.max_categories, stats.max_cat_article);
    }
    else if (strcmp(action, "search") == 0 && argc >= 4) {
        if (argc < 5) {
            printf("Usage: %s <wiki-xml-file> search \"term\" context_size\n", argv[0]);
            return 1;
        }
        
        const char* search_term = argv[3];
        int context_size = atoi(argv[4]);
        
        printf("Searching for \"%s\" with %d characters of context...\n", 
               search_term, context_size);
        
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "search-%s.txt", search_term);
        FILE* output_file = fopen(output_filename, "w");
        
        if (!output_file) {
            perror("Failed to create output file");
            return 1;
        }
        
        search_context_t ctx = {
            .search_term = search_term,
            .search_term_len = strlen(search_term),
            .context_size = context_size,
            .output_file = output_file
        };
        
        parse_wiki_xml_mt(filename, search_text_callback, &ctx, num_threads);
        
        fclose(output_file);
        printf("Search results saved to %s\n", output_filename);
    }
    else if (strcmp(action, "extract-isbns") == 0) {
        printf("Extracting ISBNs from all articles...\n");
        
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "isbns.txt");
        const char* custom_output = (argc >= 4) ? argv[3] : NULL;
        
        // If the third parameter is not a number, treat it as an output filename
        if (custom_output && !isdigit(custom_output[0])) {
            strncpy(output_filename, custom_output, sizeof(output_filename) - 1);
            output_filename[sizeof(output_filename) - 1] = '\0';
            // If there's a fourth parameter, it's the thread count
            if (argc >= 5) {
                num_threads = atoi(argv[4]);
            }
        }
        
        FILE* output_file = fopen(output_filename, "w");
        if (!output_file) {
            perror("Failed to create output file");
            return 1;
        }
        
        // Initialize the ISBN extraction context
        isbn_context_t ctx = {
            .output_file = output_file,
            .found_isbns = 0,
            .start_time = time(NULL),
            .buffer_size = 256,          // Size of each buffer entry
            .buffer_capacity = 1000      // Buffer up to 1000 ISBNs before flushing
        };
        pthread_mutex_init(&ctx.mutex, NULL);
        
        printf("Extracting ISBNs to %s...\n", output_filename);
        parse_wiki_xml_mt(filename, extract_isbns_callback, &ctx, num_threads);
        
        // Calculate performance statistics
        time_t end_time = time(NULL);
        double elapsed = difftime(end_time, ctx.start_time);
        double rate = elapsed > 0 ? ctx.found_isbns / elapsed : 0;
        
        printf("ISBN extraction complete. Found %d ISBNs in %.1f seconds (%.1f ISBNs/sec).\n", 
               ctx.found_isbns, elapsed, rate);
        
        pthread_mutex_destroy(&ctx.mutex);
        fclose(output_file);
        printf("ISBN data saved to %s\n", output_filename);
    }
    else {
        printf("Unknown action: %s\n", action);
        return 1;
    }
    
    return 0;
}
