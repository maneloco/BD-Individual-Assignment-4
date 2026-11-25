#ifndef THREAD_H
#define THREAD_H

#include <stddef.h>
#include <stdbool.h>
#include <pthread.h>

typedef struct {
    void (*function)(void *arg);
    void *arg;
} ThreadTask;

typedef struct {
    pthread_t *threads;
    size_t num_threads;

    ThreadTask *queue;
    size_t queue_capacity;
    size_t queue_size;
    size_t queue_head;
    size_t queue_tail;

    size_t active_workers;
    bool shutdown;

    pthread_mutex_t lock;
    pthread_cond_t has_work; 
    pthread_cond_t all_idle;
} ThreadPool;

ThreadPool* threadpool_create(size_t num_thread, size_t queue_capacity);
bool threadpool_add(ThreadPool *pool, void (*function)(void *arg), void *arg);
void threadpool_wait(ThreadPool *pool);
void threadpool_destroy(ThreadPool *pool);

#endif
