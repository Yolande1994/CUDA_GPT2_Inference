/* ============================================================================
   utils.h —— 各脚本共享的 C 标准库工具
   ============================================================================

   用途：
     定义一系列 xxxCheck 宏，它们调用对应的 C 标准库函数并检查返回值，
     一旦出错就打印调试信息并退出，避免在每个调用处重复写错误处理。
   提供的工具：
     1. 文件操作：fopenCheck、freadCheck、fwriteCheck、fcloseCheck、fseekCheck；
     2. 套接字关闭：scloseCheck、closesocketCheck；
     3. 内存分配：mallocCheck；
     4. token 合法性：tokenCheck；
     5. 目录与字符串辅助：create_dir_if_not_exists、find_max_step、ends_with_bin。
   ============================================================================ */
#ifndef UTILS_H
#define UTILS_H

#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
// Windows 下 dirent 的实现在 dev/unistd.h
#ifndef _WIN32
#include <dirent.h>
#include <arpa/inet.h>
#endif

// ============================================================================
// fread 便捷工具：用宏做错误检查
// 把 fopen、fread、fclose、fseek 替换为 fopenCheck、freadCheck、fcloseCheck、fseekCheck
// ============================================================================

// fopen_check —— 打开文件，失败时打印详细信息并退出
extern inline FILE *fopen_check(const char *path, const char *mode, const char *file, int line) {
    FILE *fp = fopen(path, mode);
    if (fp == NULL) {
        fprintf(stderr, "错误：无法在 %s:%d 打开文件 '%s'\n", file, line, path);
        fprintf(stderr, "错误详情：\n");
        fprintf(stderr, "  文件：%s\n", file);
        fprintf(stderr, "  行号：%d\n", line);
        fprintf(stderr, "  路径：%s\n", path);
        fprintf(stderr, "  模式：%s\n", mode);
        fprintf(stderr, "提示：请检查文件完整性\n");
        exit(EXIT_FAILURE);
    }
    return fp;
}

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)

// fread_check —— 读取文件，返回元素数不符时打印信息并退出
extern inline void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    size_t result = fread(ptr, size, nmemb, stream);
    if (result != nmemb) {
        if (feof(stream)) {
            fprintf(stderr, "错误：文件在 %s:%d 意外结束\n", file, line);
        } else if (ferror(stream)) {
            fprintf(stderr, "错误：文件读取错误，位于 %s:%d\n", file, line);
        } else {
            fprintf(stderr, "错误：部分读取，位于 %s:%d。期望 %zu 个元素，实际读取 %zu 个\n",
                    file, line, nmemb, result);
        }
        fprintf(stderr, "错误详情：\n");
        fprintf(stderr, "  文件：%s\n", file);
        fprintf(stderr, "  行号：%d\n", line);
        fprintf(stderr, "  期望元素数：%zu\n", nmemb);
        fprintf(stderr, "  已读元素数：%zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

// fclose_check —— 关闭文件，失败时打印信息并退出
extern inline void fclose_check(FILE *fp, const char *file, int line) {
    if (fclose(fp) != 0) {
        fprintf(stderr, "错误：无法在 %s:%d 关闭文件\n", file, line);
        fprintf(stderr, "错误详情：\n");
        fprintf(stderr, "  文件：%s\n", file);
        fprintf(stderr, "  行号：%d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

// sclose_check —— 关闭套接字，失败时打印信息并退出
extern inline void sclose_check(int sockfd, const char *file, int line) {
    if (close(sockfd) != 0) {
        fprintf(stderr, "错误：无法在 %s:%d 关闭套接字\n", file, line);
        fprintf(stderr, "错误详情：\n");
        fprintf(stderr, "  文件：%s\n", file);
        fprintf(stderr, "  行号：%d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
// closesocket_check —— Windows 下关闭套接字，失败时打印信息并退出
extern inline void closesocket_check(int sockfd, const char *file, int line) {
    if (closesocket(sockfd) != 0) {
        fprintf(stderr, "错误：无法在 %s:%d 关闭套接字\n", file, line);
        fprintf(stderr, "错误详情：\n");
        fprintf(stderr, "  文件：%s\n", file);
        fprintf(stderr, "  行号：%d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

// fseek_check —— 定位文件，失败时打印信息并退出
extern inline void fseek_check(FILE *fp, long off, int whence, const char *file, int line) {
    if (fseek(fp, off, whence) != 0) {
        fprintf(stderr, "错误：无法在 %s:%d 定位文件\n", file, line);
        fprintf(stderr, "错误详情：\n");
        fprintf(stderr, "  偏移：%ld\n", off);
        fprintf(stderr, "  起始位置：%d\n", whence);
        fprintf(stderr, "  文件：%s\n", file);
        fprintf(stderr, "  行号：%d\n", line);
        exit(EXIT_FAILURE);
    }
}

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

// fwrite_check —— 写入文件，返回元素数不符时打印信息并退出
extern inline void fwrite_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line) {
    size_t result = fwrite(ptr, size, nmemb, stream);
    if (result != nmemb) {
        if (feof(stream)) {
            fprintf(stderr, "错误：文件在 %s:%d 意外结束\n", file, line);
        } else if (ferror(stream)) {
            fprintf(stderr, "错误：文件写入错误，位于 %s:%d\n", file, line);
        } else {
            fprintf(stderr, "错误：部分写入，位于 %s:%d。期望 %zu 个元素，实际写入 %zu 个\n",
                    file, line, nmemb, result);
        }
        fprintf(stderr, "错误详情：\n");
        fprintf(stderr, "  文件：%s\n", file);
        fprintf(stderr, "  行号：%d\n", line);
        fprintf(stderr, "  期望元素数：%zu\n", nmemb);
        fprintf(stderr, "  已写元素数：%zu\n", result);
        exit(EXIT_FAILURE);
    }
}

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)


// ============================================================================
// malloc 错误处理封装
// ============================================================================

// malloc_check —— 分配内存，失败时打印信息并退出
extern inline void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "错误：内存分配失败，位于 %s:%d\n", file, line);
        fprintf(stderr, "错误详情：\n");
        fprintf(stderr, "  文件：%s\n", file);
        fprintf(stderr, "  行号：%d\n", line);
        fprintf(stderr, "  大小：%zu 字节\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)


// ============================================================================
// 检查所有 token 是否在合法范围内
// ============================================================================

// token_check —— 逐个检查 token 是否落在 [0, vocab_size) 内，越界则打印并退出
extern inline void token_check(const int* tokens, int token_count, int vocab_size, const char *file, int line) {
    for(int i = 0; i < token_count; i++) {
        if(!(0 <= tokens[i] && tokens[i] < vocab_size)) {
            fprintf(stderr, "错误：token 超出词表范围，位于 %s:%d\n", file, line);
            fprintf(stderr, "错误详情：\n");
            fprintf(stderr, "  文件：%s\n", file);
            fprintf(stderr, "  行号：%d\n", line);
            fprintf(stderr, "  Token：%d\n", tokens[i]);
            fprintf(stderr, "  位置：%d\n", i);
            fprintf(stderr, "  词表大小：%d\n", vocab_size);
            exit(EXIT_FAILURE);
        }
    }
}
#define tokenCheck(tokens, count, vocab) token_check(tokens, count, vocab, __FILE__, __LINE__)


// ============================================================================
// I/O 操作
// ============================================================================

// create_dir_if_not_exists —— 目录不存在时创建，创建失败则退出
extern inline void create_dir_if_not_exists(const char *dir) {
    if (dir == NULL) { return; }
    struct stat st = {0};
    if (stat(dir, &st) == -1) {
        if (mkdir(dir, 0700) == -1) {
            printf("错误：无法创建目录：%s\n", dir);
            exit(EXIT_FAILURE);
        }
        printf("已创建目录：%s\n", dir);
    }
}

// find_max_step —— 在日志目录中查找步数最大的 DONE 文件，返回该步数；无则返回 -1
extern inline int find_max_step(const char* output_log_dir) {
    if (output_log_dir == NULL) { return -1; }
    DIR* dir;
    struct dirent* entry;
    int max_step = -1;
    dir = opendir(output_log_dir);
    if (dir == NULL) { return -1; }
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "DONE_", 5) == 0) {
            int step = atoi(entry->d_name + 5);
            if (step > max_step) {
                max_step = step;
            }
        }
    }
    closedir(dir);
    return max_step;
}

// ends_with_bin —— 判断字符串是否以 ".bin" 结尾，未来可推广为通用后缀判断
extern inline int ends_with_bin(const char* str) {
    if (str == NULL) { return 0; }
    size_t len = strlen(str);
    const char* suffix = ".bin";
    size_t suffix_len = strlen(suffix);
    if (len < suffix_len) { return 0; }
    int suffix_matches = strncmp(str + len - suffix_len, suffix, suffix_len) == 0;
    return suffix_matches;
}

#endif
