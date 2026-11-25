#include "thread.h"
#include <stdlib.h>
#include <stdio.h>

static void* worker_thread(void *arg) {
    ThreadPool *pool = (ThreadPool *)arg;

    while (true) {
        pthread_mutex_lock(&pool->lock);

        while (pool->queue_size == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->has_work, &pool->lock);
        }

        if (pool->shutdown && pool->queue_size == 0) {
            pthread_mutex_unlock(&pool->lock);
            break;
        }

        ThreadTask task = pool->queue[pool->queue_head];
        pool->queue_head = (pool->queue_head + 1) % pool->queue_capacity;
        pool->queue_size--;
        pool->active_workers++;

        pthread_mutex_unlock(&pool->lock);

        task.function(task.arg);

        pthread_mutex_lock(&pool->lock);
        pool->active_workers--;

        if (pool->queue_size == 0 && pool->active_workers == 0) {
            pthread_cond_signal(&pool->all_idle);
        }

        pthread_mutex_unlock(&pool->lock);
    }

    return NULL;
}

ThreadPool* threadpool_create(size_t num_threads, size_t queue_capacity) {
    if (num_threads == 0 || queue_capacity == 0) {
        return NULL;
    }

    ThreadPool *pool = (ThreadPool *)malloc(sizeof(ThreadPool));
    if (!pool) return NULL;

    pool->num_threads = num_threads;
    pool->queue_capacity = queue_capacity;
    pool->queue_size = 0;
    pool->queue_head = 0;
    pool->queue_tail = 0;
    pool->active_workers = 0;
    pool->shutdown = false;

    pool->threads = (pthread_t *)malloc(sizeof(pthread_t) * num_threads);
    pool->queue = (ThreadTask *)malloc(sizeof(ThreadTask) * queue_capacity);

    if (!pool->threads || !pool->queue) {
        free(pool->threads);
        free(pool->queue);
        free(pool);
        return NULL;
    }

    if (pthread_mutex_init(&pool->lock, NULL) != 0 ||
        pthread_cond_init(&pool->has_work, NULL) != 0 ||
        pthread_cond_init(&pool->all_idle, NULL) != 0) {
        free(pool->threads);
        free(pool->queue);
        free(pool);
        return NULL;
    }

    for (size_t i = 0; i < num_threads; i++) {
        if (pthread_create(&pool->threads[i], NULL, worker_thread, pool) != 0) {
            threadpool_destroy(pool);
            return NULL;
        }
    }

    return pool;
}

bool threadpool_add(ThreadPool *pool, void (*function)(void *arg), void *arg) {
    if (!pool || !function) return false;

    pthread_mutex_lock(&pool->lock);

    if (pool->shutdown || pool->queue_size == pool->queue_capacity) {
        pthread_mutex_unlock(&pool->lock);
        return false;
    }

    pool->queue[pool->queue_tail].function = function;
    pool->queue[pool->queue_tail].arg = arg;
    pool->queue_tail = (pool->queue_tail + 1) % pool->queue_capacity;
    pool->queue_size++;

    pthread_cond_signal(&pool->has_work);

    pthread_mutex_unlock(&pool->lock);
    return true;
}

void threadpool_wait(ThreadPool *pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->lock);
    while (pool->queue_size > 0 || pool->active_workers > 0) {
        pthread_cond_wait(&pool->all_idle, &pool->lock);
    }
    pthread_mutex_unlock(&pool->lock);
}

void threadpool_destroy(ThreadPool *pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->lock);
    pool->shutdown = true;
    pthread_cond_broadcast(&pool->has_work);
    pthread_mutex_unlock(&pool->lock);

    for (size_t i = 0; i < pool->num_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    pthread_mutex_destroy(&pool->lock);
    pthread_cond_destroy(&pool->has_work);
    pthread_cond_destroy(&pool->all_idle);

    free(pool->threads);
    free(pool->queue);
    free(pool);
}
