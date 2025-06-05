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
#include <immintrin.h>
#include <smmintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdatomic.h>
#include <sched.h>

#ifndef likely
#  define likely(x)   __builtin_expect(!!(x), 1)
#  define unlikely(x) __builtin_expect(!!(x), 0)
#endif

/* Thread count optimization */
#ifndef OPTIMAL_THREAD_COUNT
#define OPTIMAL_THREAD_COUNT 0  // Will be determined at runtime if not specified
#endif

/* Buffer sizes */
#define WRITE_BUFFER_SIZE (64 * 1024)  // 64KB write buffer
#define THREAD_POOL_SIZE (1024 * 1024) // 1MB thread-local memory pool

/* Parser configuration */
typedef struct {
    int num_threads;
    size_t buffer_size;
    bool use_simd;
    bool enable_prefetch;
    size_t write_buffer_size;
    size_t thread_pool_size;
} parser_config_t;

/* Thread-local memory pool */
typedef struct {
    char* pool;
    size_t pool_size;
    size_t pool_used;
} thread_memory_pool_t;

static __thread thread_memory_pool_t* local_pool = NULL;

/* Cleanup stack for error handling */
typedef struct cleanup_node {
    void (*cleanup)(void*);
    void* resource;
    struct cleanup_node* next;
} cleanup_node_t;

static __thread cleanup_node_t* cleanup_stack = NULL;

/* Work item for work-stealing queue */
typedef struct work_item {
    size_t start_idx;
    size_t end_idx;
    struct work_item* next;
} work_item_t;

/* Work-stealing queue */
typedef struct {
    work_item_t* head;
    work_item_t* tail;
    pthread_mutex_t lock;
    atomic_size_t completed_items;   // <-- atomic
    size_t total_items;
    int done;
    pthread_cond_t cond;
} work_queue_t;

/* Write buffer for batch writing */
typedef struct {
    char* buffer;
    size_t capacity;
    size_t used;
    pthread_mutex_t* file_mutex;
    FILE* output_file;
} write_buffer_t;

/* Page information */
typedef struct {
    size_t start_pos;
    size_t length;
} page_info_t;

/* Callback function type */
typedef void (*article_callback)(const char* title, size_t title_len, 
                               const char* text, size_t text_len, 
                               void* user_data);

/* Thread data structure */
typedef struct {
    char* file_content;
    size_t file_size;
    article_callback callback;
    void* user_data;
    page_info_t* pages;
    size_t total_pages;
    work_queue_t* work_queue;
    int thread_id;
    size_t articles_processed;
    double start_time;
    parser_config_t* config;
} thread_data_t;

/* ISBN validation lookup table */
static const uint8_t isbn_valid_chars[256] = {
    ['0' ... '9'] = 1,
    ['-'] = 1,
    ['X'] = 1,
    ['x'] = 1
};

/* Error handling and cleanup */
void register_cleanup(void (*fn)(void*), void* resource) {
    cleanup_node_t* node = malloc(sizeof(cleanup_node_t));
    if (!node) return;
    
    node->cleanup = fn;
    node->resource = resource;
    node->next = cleanup_stack;
    cleanup_stack = node;
}

void cleanup_resources(void) {
    while (cleanup_stack) {
        cleanup_node_t* node = cleanup_stack;
        cleanup_stack = node->next;
        if (node->cleanup && node->resource) {
            node->cleanup(node->resource);
        }
        free(node);
    }
}

/* Thread-local memory pool */
void* pool_alloc(size_t size) {
    if (!local_pool) {
        local_pool = malloc(sizeof(thread_memory_pool_t));
        if (!local_pool) return NULL;
        
        local_pool->pool_size = THREAD_POOL_SIZE;
        local_pool->pool = malloc(local_pool->pool_size);
        if (!local_pool->pool) {
            free(local_pool);
            local_pool = NULL;
            return NULL;
        }
        local_pool->pool_used = 0;
    }
    
    // Align to 16 bytes for SIMD
    size = (size + 15) & ~15;
    
    if (local_pool->pool_used + size > local_pool->pool_size) {
        // Pool is full, reset
        local_pool->pool_used = 0;
    }
    
    void* ptr = local_pool->pool + local_pool->pool_used;
    local_pool->pool_used += size;
    return ptr;
}

void free_thread_pool(void* unused) {
    if (local_pool) {
        if (local_pool->pool) {
            free(local_pool->pool);
        }
        free(local_pool);
        local_pool = NULL;
    }
}

/* Work-stealing queue implementation */
work_queue_t* create_work_queue(void) {
    work_queue_t* queue = malloc(sizeof(work_queue_t));
    if (!queue) return NULL;
    
    queue->head = NULL;
    queue->tail = NULL;
    queue->done = 0;
    queue->total_items = 0;
    queue->completed_items = 0;
    pthread_mutex_init(&queue->lock, NULL);
    pthread_cond_init(&queue->cond, NULL);
    
    return queue;
}

void destroy_work_queue(work_queue_t* queue) {
    if (!queue) return;
    
    pthread_mutex_destroy(&queue->lock);
    pthread_cond_destroy(&queue->cond);
    
    // Free remaining work items
    work_item_t* item = queue->head;
    while (item) {
        work_item_t* next = item->next;
        free(item);
        item = next;
    }
    
    free(queue);
}

void add_work(work_queue_t* queue, size_t start, size_t end) {
    work_item_t* item = malloc(sizeof(work_item_t));
    if (!item) return;
    
    item->start_idx = start;
    item->end_idx = end;
    item->next = NULL;
    
    pthread_mutex_lock(&queue->lock);
    
    if (queue->tail) {
        queue->tail->next = item;
    } else {
        queue->head = item;
    }
    queue->tail = item;
    queue->total_items++;
    
    pthread_cond_signal(&queue->cond);
    pthread_mutex_unlock(&queue->lock);
}

work_item_t* steal_work(work_queue_t* queue) {
    pthread_mutex_lock(&queue->lock);
    
    while (!queue->head && !queue->done) {
        pthread_cond_wait(&queue->cond, &queue->lock);
    }
    
    work_item_t* item = queue->head;
    if (item) {
        queue->head = item->next;
        if (!queue->head) {
            queue->tail = NULL;
        }
    }
    
    pthread_mutex_unlock(&queue->lock);
    return item;
}

static void mark_work_done(work_queue_t* q)
{
    pthread_mutex_lock(&q->lock);
    q->completed_items++;
    if (q->completed_items == q->total_items)
        pthread_cond_broadcast(&q->cond);   // wake waiting producer
    pthread_mutex_unlock(&q->lock);
}

/* Boyer-Moore-Horspool algorithm */
void prepare_bad_char_table(const char* pattern, size_t pattern_len, size_t bad_char[256]) {
    for (size_t i = 0; i < 256; i++) {
        bad_char[i] = pattern_len;
    }
    
    for (size_t i = 0; i < pattern_len - 1; i++) {
        bad_char[(unsigned char)pattern[i]] = pattern_len - 1 - i;
    }
}

/* SSE4.2 optimized string search for short patterns */
const char* simd_find_pattern(const char* haystack, size_t len, const char* pattern, size_t pattern_len) {
    if (pattern_len <= 16 && pattern_len > 0 && len >= pattern_len) {
        __m128i first = _mm_set1_epi8(pattern[0]);
        
        size_t i;
        for (i = 0; i + 16 <= len; i += 16) {
            __m128i chunk = _mm_loadu_si128((const __m128i*)(haystack + i));
            __m128i eq_first = _mm_cmpeq_epi8(chunk, first);
            int mask = _mm_movemask_epi8(eq_first);
            
            while (mask) {
                int pos = __builtin_ctz(mask);
                if (i + pos + pattern_len <= len && 
                    memcmp(haystack + i + pos, pattern, pattern_len) == 0) {
                    return haystack + i + pos;
                }
                mask &= mask - 1;
            }
        }
        
        // Check remaining bytes
        for (; i + pattern_len <= len; i++) {
            if (memcmp(haystack + i, pattern, pattern_len) == 0) {
                return haystack + i;
            }
        }
    }
    return NULL;
}

/* Fast string search using Boyer-Moore-Horspool algorithm */
char* fast_strstr(const char* haystack,
                  const char* needle,
                  size_t hay_len,
                  size_t nee_len)
{
    /* trivial cases -------------------------------------------------------- */
    if (nee_len == 0 || nee_len > hay_len) return NULL;
    if (nee_len == 1)
        return (char*)memchr(haystack, needle[0], hay_len);

#if 0   /* AVX2 block still disabled until re-verified */
    /* … */
#endif

    /* SIMD helper for very short needles (≤16) ----------------------------- */
    if (nee_len <= 16)
        return (char*)simd_find_pattern(haystack, hay_len,
                                        needle, nee_len);

    /* Boyer–Moore–Horspool fallback --------------------------------------- */
    size_t bad[256];
    prepare_bad_char_table(needle, nee_len, bad);

    size_t i = nee_len - 1;
    while (i < hay_len) {
        size_t j = nee_len - 1;
        const char* hptr = haystack + i;
        while (haystack[i - (nee_len - 1 - j)] == needle[j]) {
            if (j-- == 0)
                return (char*)(haystack + i - (nee_len - 1));
        }
        size_t shift = bad[(unsigned char)haystack[i]];
        size_t diff  = nee_len - j;
        i += (shift > diff) ? shift : diff;
    }
    return NULL;
}

/* Calculate ISBN-10 checksum */
int calculate_isbn10_checksum(const char* start, const char* end) {
    int sum = 0;
    int weight = 10;
    
    for (const char* p = start; p < end && weight > 1; p++) {
        if (*p >= '0' && *p <= '9') {
            sum += (*p - '0') * weight;
            weight--;
        }
    }
    
    int check = (11 - (sum % 11)) % 11;
    return check;
}

/* Calculate ISBN-13 checksum */
int calculate_isbn13_checksum(const char* start, const char* end) {
    int sum = 0;
    int position = 0;
    
    for (const char* p = start; p < end - 1; p++) {
        if (*p >= '0' && *p <= '9') {
            int digit = *p - '0';
            sum += digit * ((position % 2 == 0) ? 1 : 3);
            position++;
        }
    }
    
    int check = (10 - (sum % 10)) % 10;
    return check;
}

/* Validate ISBN format and checksum */
int validate_isbn_fast(const char* start, const char* end, int* checksum_valid) {
    size_t len = end - start;
    if (len < 9 || len > 17) return 0;
    
    int digits = 0;
    char last_char = 0;
    
    // Count digits and get last character
    for (const char* p = start; p < end; p++) {
        if (*p >= '0' && *p <= '9') {
            digits++;
            last_char = *p;
        } else if ((*p == 'X' || *p == 'x') && p == end - 1) {
            last_char = 'X';
        }
    }
    
    *checksum_valid = 1; // Assume valid unless proven otherwise
    
    // ISBN-10: 9 digits + checksum (digit or X)
    if ((digits == 10) || (digits == 9 && last_char == 'X')) {
        // Validate checksum
        int calculated_check = calculate_isbn10_checksum(start, end);
        int actual_check = (last_char == 'X') ? 10 : (last_char - '0');
        *checksum_valid = (calculated_check == actual_check);
        return 1; // Valid ISBN-10 format
    }
    
    // ISBN-13: 13 digits
    if (digits == 13) {
        // Validate checksum
        int calculated_check = calculate_isbn13_checksum(start, end);
        int actual_check = last_char - '0';
        *checksum_valid = (calculated_check == actual_check);
        return 1; // Valid ISBN-13 format
    }
    
    return 0; // Invalid format
}

/* Buffered write implementation */
write_buffer_t* create_write_buffer(FILE* file, pthread_mutex_t* mutex, size_t capacity) {
    write_buffer_t* wb = malloc(sizeof(write_buffer_t));
    if (!wb) return NULL;
    
    wb->buffer = malloc(capacity);
    if (!wb->buffer) {
        free(wb);
        return NULL;
    }
    
    wb->capacity = capacity;
    wb->used = 0;
    wb->output_file = file;
    wb->file_mutex = mutex;
    
    return wb;
}

/* forward declarations ─ needed by flush_and_free_write_buffer */
void flush_write_buffer(write_buffer_t* wb);
void destroy_write_buffer(write_buffer_t* wb);
/* helper that will be used as cleanup handler -------------- */
static void flush_and_free_write_buffer(void* res)
{
    write_buffer_t* wb = (write_buffer_t*)res;
    if (wb) {
        flush_write_buffer(wb);
        destroy_write_buffer(wb);
    }
}

/* replace body of existing flush_write_buffer -------------- */
void flush_write_buffer(write_buffer_t* wb)
{
    if (!wb || wb->used == 0) return;

    pthread_mutex_lock(wb->file_mutex);
    size_t written = fwrite(wb->buffer, 1, wb->used, wb->output_file);
    pthread_mutex_unlock(wb->file_mutex);

    if (written != wb->used)
        perror("Short write");

    wb->used = 0;
}
// ---------------------------------------------------------------------


void buffered_write(write_buffer_t* wb, const char* data, size_t len) {
    if (wb->used + len > wb->capacity) {
        flush_write_buffer(wb);
    }
    
    memcpy(wb->buffer + wb->used, data, len);
    wb->used += len;
}

void destroy_write_buffer(write_buffer_t* wb) {
    if (!wb) return;
    
    flush_write_buffer(wb);
    
    if (wb->buffer) {
        free(wb->buffer);
    }
    free(wb);
}

/* Extract title and text from page */
void extract_title_and_text(const char* page_start, size_t page_len,
                         char** title_start, size_t* title_len,
                         char** text_start, size_t* text_len) {
    
    const char* title_start_tag = "<title>";
    const size_t title_start_len = strlen(title_start_tag);
    const char* title_end_tag = "</title>";
    const size_t title_end_len = strlen(title_end_tag);
    const char* text_start_tag = "<text";
    const size_t text_start_len = strlen(text_start_tag);
    const char* text_end_tag = "</text>";
    const size_t text_end_len = strlen(text_end_tag);
    
    *title_start = NULL;
    *title_len = 0;
    *text_start = NULL;
    *text_len = 0;
    
    char* temp_title_start = fast_strstr(page_start, title_start_tag, page_len, title_start_len);
    if (!temp_title_start) return;
    
    temp_title_start += title_start_len;
    size_t remaining = page_start + page_len - temp_title_start;
    char* title_end = fast_strstr(temp_title_start, title_end_tag, remaining, title_end_len);
    if (!title_end) return;
    
    *title_start = temp_title_start;
    *title_len = title_end - temp_title_start;
    
    char* temp_text_start = fast_strstr(title_end, text_start_tag, page_start + page_len - title_end, text_start_len);
    if (!temp_text_start) return;
    
    char* text_content_start = strchr(temp_text_start, '>');
    if (!text_content_start) return;
    text_content_start++;
    
    remaining = page_start + page_len - text_content_start;
    char* text_end = fast_strstr(text_content_start, text_end_tag, remaining, text_end_len);
    if (!text_end) return;
    
    *text_start = text_content_start;
    *text_len = text_end - text_content_start;
}

/* Process single page with prefetching */
void process_single_page(page_info_t* page, thread_data_t* data, page_info_t* next_page) {
    char* page_start = data->file_content + page->start_pos;
    
    // Prefetch next page if available and prefetching is enabled
    if (next_page && data->config->enable_prefetch) {
        __builtin_prefetch(&next_page->start_pos, 0, 3);
        __builtin_prefetch(data->file_content + next_page->start_pos, 0, 1);
    }
    
    char* title_start;
    size_t title_len;
    char* text_start;
    size_t text_len;
    
    extract_title_and_text(page_start, page->length,
                          &title_start, &title_len,
                          &text_start, &text_len);
    
    if (title_start && text_start) {
        data->callback(title_start, title_len, text_start, text_len, data->user_data);
    }
}

/* Worker thread function with work stealing */
void* process_pages_worker(void* arg) {
    thread_data_t* data = (thread_data_t*)arg;
    size_t processed = 0;
    
    // Register cleanup for thread-local pool
    register_cleanup(free_thread_pool, NULL);
    register_cleanup(flush_and_free_write_buffer, NULL);   // handles TLS WB
    
    printf("Thread %d: started\n", data->thread_id);
    
    work_item_t* work;
    while ((work = steal_work(data->work_queue)) != NULL) {
        // Process the work item
        for (size_t i = work->start_idx; i < work->end_idx; i++) {
            page_info_t* next_page = (i + 1 < work->end_idx) ? &data->pages[i + 1] : NULL;
            process_single_page(&data->pages[i], data, next_page);
            processed++;
            
            // Report progress periodically
            if (processed % 10000 == 0) {
                double elapsed = (double)(time(NULL) - data->start_time);
                double rate = (elapsed > 0) ? processed / elapsed : 0;
                printf("Thread %d: Processed %zu articles, %.1f articles/sec\n", 
                       data->thread_id, processed, rate);
            }

            // --- quiet logging inside worker -------------------------------------------
            if (unlikely(processed % 100000 == 0 && data->thread_id == 0)) {
                double elapsed = difftime(time(NULL), data->start_time);
                printf("Progress: %zu articles, %.1f/sec\n",
                       processed, processed / (elapsed > 0 ? elapsed : 1));
            }
        }
        
        mark_work_done(data->work_queue);
        free(work);
    }
    
    data->articles_processed = processed;
    
    // Cleanup thread-local resources
    cleanup_resources();
    
    return NULL;
}

/* Find all pages in the file */
size_t find_all_pages(char* file_content, size_t file_size, page_info_t** pages_out) {
    const char* page_start_tag = "<page>";
    const size_t page_start_len = strlen(page_start_tag);
    const char* page_end_tag = "</page>";
    const size_t page_end_len = strlen(page_end_tag);
    
    size_t max_pages = 1000000;
    size_t found_pages = 0;
    page_info_t* pages = malloc(max_pages * sizeof(page_info_t));
    if (!pages) {
        perror("Failed to allocate memory for pages");
        return 0;
    }
    
    printf("First pass: locating all pages...\n");
    
    char* current = file_content;
    char* end = file_content + file_size;
    size_t report_threshold = 1000000;
    clock_t start_time = clock();
    
    while (current < end) {
        size_t remaining = end - current;
        
        char* page_start = fast_strstr(current, page_start_tag, remaining, page_start_len);
        if (!page_start) break;
        
        remaining = end - page_start;
        char* page_end = fast_strstr(page_start, page_end_tag, remaining, page_end_len);
        if (!page_end) break;
        
        page_end += page_end_len;
        
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
    
    double elapsed = (double)(clock() - start_time) / CLOCKS_PER_SEC;
    printf("Found %zu pages in %.1f seconds\n", found_pages, elapsed);
    
    *pages_out = pages;
    return found_pages;
}

/* Multi-threaded Wikipedia XML parser */
int parse_wiki_xml_mt(const char* filename, article_callback callback, void* user_data, parser_config_t* config) {
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
    
    // --- I/O hints and huge-page advice (inside parse_wiki_xml_mt(), after open)-
#ifdef O_NOATIME
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NOATIME);
#endif
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    
    int num_threads = config->num_threads;
    if (OPTIMAL_THREAD_COUNT > 0) {
        num_threads = OPTIMAL_THREAD_COUNT;
    } else if (num_threads <= 0) {
        num_threads = sysconf(_SC_NPROCESSORS_ONLN);
        if (num_threads <= 0) num_threads = 4;
    }
    
    printf("Using %d threads with work-stealing\n", num_threads);
    if (config->use_simd) printf("SIMD optimizations enabled\n");
    if (config->enable_prefetch) printf("Prefetching enabled\n");
    
    // Find all pages
    page_info_t* pages = NULL;
    size_t total_pages = find_all_pages(file_content, sb.st_size, &pages);
    
    if (total_pages == 0) {
        munmap(file_content, sb.st_size);
        close(fd);
        return 1;
    }
    
    // Create work queue
    work_queue_t* work_queue = create_work_queue();
    if (!work_queue) {
        free(pages);
        munmap(file_content, sb.st_size);
        close(fd);
        return 1;
    }
    
    // Divide work into chunks
    size_t chunk_size = 1000; // Process 1000 pages per work item
    for (size_t i = 0; i < total_pages; i += chunk_size) {
        size_t end = (i + chunk_size < total_pages) ? i + chunk_size : total_pages;
        add_work(work_queue, i, end);
    }
    
    pthread_t* threads = malloc(num_threads * sizeof(pthread_t));
    thread_data_t* thread_data = malloc(num_threads * sizeof(thread_data_t));
    
    time_t start_time = time(NULL);
    
    // Start worker threads
    for (int i = 0; i < num_threads; i++) {
        thread_data[i].file_content = file_content;
        thread_data[i].file_size = sb.st_size;
        thread_data[i].callback = callback;
        thread_data[i].user_data = user_data;
        thread_data[i].pages = pages;
        thread_data[i].total_pages = total_pages;
        thread_data[i].work_queue = work_queue;
        thread_data[i].thread_id = i;
        thread_data[i].articles_processed = 0;
        thread_data[i].start_time = start_time;
        thread_data[i].config = config;
        
        if (pthread_create(&threads[i], NULL, process_pages_worker, &thread_data[i]) != 0) {
            perror("Failed to create thread");
            // Cleanup and return
            work_queue->done = 1;
            pthread_cond_broadcast(&work_queue->cond);
            for (int j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
            free(threads);
            free(thread_data);
            destroy_work_queue(work_queue);
            free(pages);
            munmap(file_content, sb.st_size);
            close(fd);
            return 1;
        }
    }
    
    /* Wait for all work to complete */
    pthread_mutex_lock(&work_queue->lock);
    while (atomic_load_explicit(&work_queue->completed_items,
                                memory_order_acquire) < work_queue->total_items)
        pthread_cond_wait(&work_queue->cond, &work_queue->lock);
    work_queue->done = 1;
    pthread_cond_broadcast(&work_queue->cond);
    pthread_mutex_unlock(&work_queue->lock);
    
    // Wait for threads to finish
    size_t total_articles_processed = 0;
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
        printf("Thread %d complete: processed %zu articles\n", i, thread_data[i].articles_processed);
        total_articles_processed += thread_data[i].articles_processed;
    }
    
    double elapsed = difftime(time(NULL), start_time);
    elapsed = (elapsed == 0) ? 0.1 : elapsed;
    double rate = total_articles_processed / elapsed;
    
    printf("Processing complete. Processed %zu articles in %.1f seconds (%.1f articles/sec)\n", 
           total_articles_processed, elapsed, rate);
    
    // Cleanup
    free(threads);
    free(thread_data);
    destroy_work_queue(work_queue);
    free(pages);
    munmap(file_content, sb.st_size);
    close(fd);
    
    return 0;
}

/* Category counting implementation */
typedef struct {
    size_t total_articles;
    size_t articles_with_cats;
    size_t total_categories;
    size_t max_categories;
    char max_cat_article[256];
    pthread_mutex_t mutex;
} category_stats_t;

void count_categories_callback(const char* title, size_t title_len, 
                             const char* text, size_t text_len, 
                             void* user_data) {
    category_stats_t* stats = (category_stats_t*)user_data;
    
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
    
    pthread_mutex_lock(&stats->mutex);
    stats->total_articles++;
    
    if (categories > 0) {
        stats->articles_with_cats++;
        stats->total_categories += categories;
        
        if (categories > stats->max_categories) {
            stats->max_categories = categories;
            
            size_t copy_len = title_len < 255 ? title_len : 255;
            memcpy(stats->max_cat_article, title, copy_len);
            stats->max_cat_article[copy_len] = '\0';
        }
    }
    pthread_mutex_unlock(&stats->mutex);
    
    // --- conditional printing in callbacks (example for category counter) ------
    if (unlikely(stats->total_articles % 500000 == 0))   /* every 0.5 M */
        fprintf(stderr, "Scanned %zu articles so far\n", stats->total_articles);
}

/* Search implementation */
typedef struct {
    const char* search_term;
    size_t search_term_len;
    int context_size;
    FILE* output_file;
    pthread_mutex_t mutex;
    write_buffer_t* write_buffer;
} search_context_t;

void search_text_callback(const char* title, size_t title_len, 
                         const char* text, size_t text_len, 
                         void* user_data) {
    search_context_t* ctx = (search_context_t*)user_data;
    
    // Get thread-local write buffer
    static __thread write_buffer_t* local_write_buffer = NULL;
    if (!local_write_buffer) {
        local_write_buffer =
            create_write_buffer(ctx->output_file, &ctx->mutex, WRITE_BUFFER_SIZE);
        if (local_write_buffer)
            register_cleanup(flush_and_free_write_buffer, local_write_buffer);
    }
    
    const char* pos = text;
    const char* text_end = text + text_len;
    
    while (pos < text_end) {
        size_t remaining = text_end - pos;
        const char* match = fast_strstr(pos, ctx->search_term, remaining, ctx->search_term_len);
        if (!match) break;
        
        char buffer[4096];
        int written = snprintf(buffer, sizeof(buffer), "Article: %.*s\n", (int)title_len, title);
        
        const char* context_start = match - ctx->context_size;
        if (context_start < text) context_start = text;
        
        const char* context_end = match + ctx->search_term_len + ctx->context_size;
        if (context_end > text_end) context_end = text_end;
        
        written += snprintf(buffer + written, sizeof(buffer) - written, 
                           "...%.*s...\n\n", (int)(context_end - context_start), context_start);
        
        buffered_write(local_write_buffer, buffer, written);
        
        pos = match + 1;
    }
}

/* ISBN extraction implementation - simplified structures */
typedef struct {
    FILE* output_file;
    pthread_mutex_t mutex;
    int found_isbns;
    time_t start_time;
    int buffer_size;
    int buffer_capacity;
    write_buffer_t* write_buffer;
} isbn_context_t;

/* ISBN pattern handlers - simplified without function pointers */
typedef struct {
    const char* prefix;
    size_t prefix_len;
} isbn_pattern_t;

static const isbn_pattern_t isbn_patterns[] = {
    {"ISBN ", 5},
    {"ISBN:", 5},
    {"ISBN-10:", 8},
    {"ISBN-13:", 8},
    {"ISBN-10 ", 8},
    {"ISBN-13 ", 8},
    {"{{ISBN|", 7},
    {"{{BibISBN|", 10},
    {"ISBN=", 5},
    {"isbn=", 5},
    {"ISBN&nbsp;", 10},
    {NULL, 0}
};

static inline int is_valid_isbn_char_fast(char c);

/*
 * focused_isbn_extraction_callback - Dedicated callback for optimized ISBN extraction.
 * This function is invoked by parse_wiki_xml_mt when the 'extract-isbns' action is selected.
 * It contains refined logic for finding and validating ISBNs efficiently.
 */
void focused_isbn_extraction_callback(const char* title, size_t title_len,
                                   const char* text, size_t text_len,
                                   void* user_data) {
    isbn_context_t* ctx = (isbn_context_t*)user_data;

    // Thread-local write buffer
    static __thread write_buffer_t* local_write_buffer = NULL;
    if (!local_write_buffer) {
        local_write_buffer =
            create_write_buffer(ctx->output_file, &ctx->mutex, WRITE_BUFFER_SIZE);
        if (local_write_buffer)
            register_cleanup(flush_and_free_write_buffer, local_write_buffer);
    }

    // Note: Title is directly available from the arguments (title, title_len)
    // No need to call extract_title_and_text again if it's already provided
    // by the main parsing loop.

    const char* text_end = text + text_len;
    const char* current_search_pos = text;

    // Optimized search: Find the earliest occurrence of any ISBN pattern in a single pass
    // over the text for each found ISBN.
    while (current_search_pos < text_end) {
        const char* earliest_match_overall = NULL;
        const isbn_pattern_t* matched_pattern_info = NULL;

        // Iterate through all defined ISBN patterns
        for (const isbn_pattern_t* pattern = isbn_patterns; pattern->prefix; pattern++) {
            if (current_search_pos >= text_end) break; // Optimization: if we are at or past the end

            size_t remaining_text_len = text_end - current_search_pos;
            // Ensure remaining_text_len is not smaller than pattern->prefix_len before search
            if (remaining_text_len < pattern->prefix_len) {
                continue;
            }
            const char* current_pattern_match = fast_strstr(current_search_pos, pattern->prefix, remaining_text_len, pattern->prefix_len);

            if (current_pattern_match) {
                if (!earliest_match_overall || current_pattern_match < earliest_match_overall) {
                    earliest_match_overall = current_pattern_match;
                    matched_pattern_info = pattern;
                }
            }
        }

        if (!earliest_match_overall) {
            break; // No more ISBN patterns found in the rest of the text
        }

        // At this point, earliest_match_overall points to the start of the found pattern prefix
        // and matched_pattern_info has the details of which pattern was found.
        // Proceed to extract the ISBN value itself, validate it, and record it.
        const char* isbn_start = earliest_match_overall + matched_pattern_info->prefix_len;

        // Skip whitespace (same as before)
        while (isbn_start < text_end && (*isbn_start == ' ' || *isbn_start == ':' || *isbn_start == '=')) {
            isbn_start++;
        }

        const char* isbn_end = isbn_start;
        // Find end based on pattern type (same as before)
        if (matched_pattern_info == &isbn_patterns[6] || matched_pattern_info == &isbn_patterns[7]) { // Template patterns like {{ISBN|...}}
            while (isbn_end < text_end && *isbn_end != '|' && *isbn_end != '}') {
                isbn_end++;
            }
        } else { // Regular ISBN patterns
            isbn_end = isbn_start; // Start scan from the beginning of potential ISBN data
            while (isbn_end < text_end && (is_valid_isbn_char_fast(*isbn_end) || *isbn_end == ' ')) {
                isbn_end++;
            }
            // Trim trailing spaces and hyphens (same as before)
            while (isbn_end > isbn_start && (*(isbn_end-1) == ' ' || *(isbn_end-1) == '-')) {
                isbn_end--;
            }
        }

        if (isbn_start < isbn_end) { // Ensure there's a non-empty potential ISBN string
            int checksum_valid = 1; // Assume valid unless check fails
            if (validate_isbn_fast(isbn_start, isbn_end, &checksum_valid)) {
                char entry[512];
                int len;

                if (!checksum_valid) {
                    len = snprintf(entry, sizeof(entry), "%.*s|%.*s|CHECKSUM_WRONG\n",
                                  (int)title_len, title, (int)(isbn_end - isbn_start), isbn_start);
                } else {
                    len = snprintf(entry, sizeof(entry), "%.*s|%.*s\n",
                                  (int)title_len, title, (int)(isbn_end - isbn_start), isbn_start);
                }

                if (len > 0 && len < sizeof(entry)) {
                    buffered_write(local_write_buffer, entry, len);

                    pthread_mutex_lock(&ctx->mutex);
                    ctx->found_isbns++;
                    if (ctx->found_isbns % 10000 == 0) {
                        printf("Found %d ISBNs so far\n", ctx->found_isbns);
                    }
                    pthread_mutex_unlock(&ctx->mutex);
                }
            }
        }

        // Advance current_search_pos for the next iteration.
        // Important: Advance beyond the *start* of the found pattern prefix.
        current_search_pos = earliest_match_overall + 1;
    }
}


void extract_isbns_callback(const char* title, size_t title_len, 
                           const char* text, size_t text_len, 
                           void* user_data) {
    isbn_context_t* ctx = (isbn_context_t*)user_data;
    
    // No longer use isbn_buffer_t, directly write to write_buffer
    static __thread write_buffer_t* local_write_buffer = NULL;
    if (!local_write_buffer) {
        local_write_buffer =
            create_write_buffer(ctx->output_file, &ctx->mutex, WRITE_BUFFER_SIZE);
        if (local_write_buffer)
            register_cleanup(flush_and_free_write_buffer, local_write_buffer);
    }
    
    const char* pos = text;
    const char* text_end = text + text_len;
    
    // Process with pattern matching
    while (pos < text_end) {
        const char* closest_match = NULL;
        const isbn_pattern_t* closest_pattern = NULL;
        
        // Find closest pattern
        for (const isbn_pattern_t* pattern = isbn_patterns; pattern->prefix; pattern++) {
            size_t remaining = text_end - pos;
            const char* match = fast_strstr(pos, pattern->prefix, remaining, pattern->prefix_len);
            
            if (match && (!closest_match || match < closest_match)) {
                closest_match = match;
                closest_pattern = pattern;
            }
        }
        
        if (!closest_match) break;
        
        // Extract ISBN using appropriate handler
        const char* isbn_start = closest_match + closest_pattern->prefix_len;
        
        // Simple inline extraction instead of calling function pointer
        const char* isbn_end = isbn_start;
        
        // Skip whitespace
        while (isbn_start < text_end && (*isbn_start == ' ' || *isbn_start == ':' || *isbn_start == '=')) {
            isbn_start++;
        }
        
        // Find end based on pattern type
        if (closest_pattern == &isbn_patterns[6] || closest_pattern == &isbn_patterns[7]) { // Template patterns
            while (isbn_end < text_end && *isbn_end != '|' && *isbn_end != '}') {
                isbn_end++;
            }
        } else {
            isbn_end = isbn_start;
            while (isbn_end < text_end && (is_valid_isbn_char_fast(*isbn_end) || *isbn_end == ' ')) {
                isbn_end++;
            }
            // Trim trailing
            while (isbn_end > isbn_start && (*(isbn_end-1) == ' ' || *(isbn_end-1) == '-')) {
                isbn_end--;
            }
        }
        
        if (isbn_start < isbn_end) {
            int checksum_valid = 1;
            if (validate_isbn_fast(isbn_start, isbn_end, &checksum_valid)) {
                char entry[512];
                int len;
                
                if (!checksum_valid) {
                    len = snprintf(entry, sizeof(entry), "%.*s|%.*s|CHECKSUM_WRONG\n", 
                                  (int)title_len, title, (int)(isbn_end - isbn_start), isbn_start);
                } else {
                    len = snprintf(entry, sizeof(entry), "%.*s|%.*s\n", 
                                  (int)title_len, title, (int)(isbn_end - isbn_start), isbn_start);
                }
                
                if (len > 0 && len < sizeof(entry)) {
                    buffered_write(local_write_buffer, entry, len);
                    
                    pthread_mutex_lock(&ctx->mutex);
                    ctx->found_isbns++;
                    if (ctx->found_isbns % 10000 == 0) {
                        printf("Found %d ISBNs so far\n", ctx->found_isbns);
                    }
                    pthread_mutex_unlock(&ctx->mutex);
                }
            }
        }
        
        pos = closest_match + 1;
    }
}

static inline int is_valid_isbn_char_fast(char c) {
    return (c >= '0' && c <= '9') || c == '-' || c == 'X' || c == 'x';
}

/* Configuration loading */
parser_config_t* create_default_config(void) {
    parser_config_t* config = malloc(sizeof(parser_config_t));
    if (!config) return NULL;
    
    config->num_threads = 0;  // Auto-detect
    config->buffer_size = 256;
    config->use_simd = true;
    config->enable_prefetch = true;
    config->write_buffer_size = WRITE_BUFFER_SIZE;
    config->thread_pool_size = THREAD_POOL_SIZE;
    
    return config;
}

/* Main function */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <wiki-xml-file> [action] [options]\n", argv[0]);
        printf("Actions:\n");
        printf("  count-cats         Count categories in articles (default)\n");
        printf("  search \"term\" N    Search for a term with N characters of context\n");
        printf("  extract-isbns      Extract all ISBNs with article titles\n");
        printf("Options:\n");
        printf("  -t N               Use N threads (default: auto-detect)\n");
        printf("  --no-simd          Disable SIMD optimizations\n");
        printf("  --no-prefetch      Disable prefetching\n");
        return 1;
    }
    
    const char* filename = argv[1];
    const char* action = (argc >= 3) ? argv[2] : "count-cats";
    
    // Create default configuration
    parser_config_t* config = create_default_config();
    if (!config) {
        fprintf(stderr, "Failed to create configuration\n");
        return 1;
    }
    
    // Argument parsing needs to be aware of the action
    int opt_idx = 3; // Starting index for options, assuming argv[0]=program, argv[1]=file, argv[2]=action

    if (strcmp(action, "extract-isbns") == 0) {
        // For extract-isbns, argv[3] could be an output file or an option
        if (argc > 3 && argv[3][0] != '-' && !isdigit(argv[3][0])) {
            // This looks like an output file path
            opt_idx = 4; // Options start after the output file path
        }
    } else if (strcmp(action, "search") == 0) {
        // For search, argv[3] is search_term, argv[4] is context_size
        opt_idx = 5; // Options start after search term and context size
    }
    // For count-cats, options start at argv[3]

    // Parse additional options
    for (int i = opt_idx; i < argc; i++) {
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            config->num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-simd") == 0) {
            config->use_simd = false;
        } else if (strcmp(argv[i], "--no-prefetch") == 0) {
            config->enable_prefetch = false;
        }
        // Add other general options here if needed
    }
    
    if (strcmp(action, "count-cats") == 0) {
        printf("Counting categories in Wikipedia articles...\n");
        
        category_stats_t stats = {0};
        pthread_mutex_init(&stats.mutex, NULL);
        
        parse_wiki_xml_mt(filename, count_categories_callback, &stats, config);
        
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
        
        pthread_mutex_destroy(&stats.mutex);
    }
    else if (strcmp(action, "search") == 0 && argc >= 5) {
        const char* search_term = argv[3];
        int context_size = atoi(argv[4]);
        
        printf("Searching for \"%s\" with %d characters of context...\n", 
               search_term, context_size);
        
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "search-%s.txt", search_term);
        FILE* output_file = fopen(output_filename, "w");
        
        if (!output_file) {
            perror("Failed to create output file");
            free(config);
            return 1;
        }
        
        search_context_t ctx = {
            .search_term = search_term,
            .search_term_len = strlen(search_term),
            .context_size = context_size,
            .output_file = output_file,
            .write_buffer = NULL
        };
        pthread_mutex_init(&ctx.mutex, NULL);
        
        parse_wiki_xml_mt(filename, search_text_callback, &ctx, config);
        
        // Flush any remaining buffers
        // (handled by thread cleanup)
        
        pthread_mutex_destroy(&ctx.mutex);
        fclose(output_file);
        printf("Search results saved to %s\n", output_filename);
    }
    else if (strcmp(action, "extract-isbns") == 0) {
        // For detailed performance analysis of ISBN extraction, consider using profiling tools
        // like 'perf' on Linux. For example:
        // perf record -g ./wiki_parser <path_to_xml_dump> extract-isbns
        // perf report
        // This can help identify specific hotspots in the code.
        printf("Extracting ISBNs from all articles...\n");
        
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "isbns.txt"); // Default output filename
        
        // Simplified argument parsing for extract-isbns
        // Primary arguments: input file (argv[1]), optional output file (argv[3] if not an option)
        if (argc > 3 && argv[3][0] != '-' && !isdigit(argv[3][0])) { // Check if argv[3] is likely a filename
            strncpy(output_filename, argv[3], sizeof(output_filename) - 1);
            output_filename[sizeof(output_filename) - 1] = '\0';
            // Options will be parsed from argv[4] onwards by the generic option parser
        }
        // Else, options are parsed from argv[3] onwards

        FILE* output_file = fopen(output_filename, "w");
        if (!output_file) {
            perror("Failed to create output file");
            free(config);
            return 1;
        }
        
        isbn_context_t ctx = {
            .output_file = output_file,
            .found_isbns = 0,
            .start_time = time(NULL),
            .buffer_size = 512,
            .buffer_capacity = 1000,
            .write_buffer = NULL
        };
        pthread_mutex_init(&ctx.mutex, NULL);
        
        printf("Extracting ISBNs to %s...\n", output_filename);
        
        // Add timing for parse_wiki_xml_mt
        clock_t parsing_start_time = clock();
        time_t wall_parsing_start_time = time(NULL);

        // Use the new callback function for ISBN extraction
        parse_wiki_xml_mt(filename, focused_isbn_extraction_callback, &ctx, config);

        clock_t parsing_end_time = clock();
        time_t wall_parsing_end_time = time(NULL);
        double parsing_cpu_time_used = ((double) (parsing_end_time - parsing_start_time)) / CLOCKS_PER_SEC;
        double parsing_wall_time_used = difftime(wall_parsing_end_time, wall_parsing_start_time);

        printf("Core parsing and ISBN extraction phase completed in %.2f seconds (CPU time) / %.2f seconds (wall clock time).\n",
               parsing_cpu_time_used, parsing_wall_time_used);

        time_t end_time = time(NULL); // This is overall end time for the action
        double elapsed = difftime(end_time, ctx.start_time); // Overall wall time for the action
        double rate = elapsed > 0 ? ctx.found_isbns / elapsed : 0;
        
        printf("ISBN extraction complete. Found %d ISBNs in %.1f seconds (%.1f ISBNs/sec).\n", 
               ctx.found_isbns, elapsed, rate);
        
        pthread_mutex_destroy(&ctx.mutex);
        fclose(output_file);
        printf("ISBN data saved to %s\n", output_filename);
    }
    else {
        printf("Unknown action: %s\n", action);
        free(config);
        return 1;
    }
    
    free(config);
    return 0;
}
