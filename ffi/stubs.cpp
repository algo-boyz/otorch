#include <stddef.h>
#include <stdint.h>

extern "C" {
    // Dummy implementations for Rust Stream callbacks
    // We don't need these because we aren't using Rust

    void tch_write_stream_destructor(void* ptr) {}
    void tch_write_stream_write(void* ptr, const void* data, size_t len) {}

    void tch_read_stream_destructor(void* ptr) {}
    int tch_read_stream_read(void* ptr, void* buf, size_t len, const char* what) { return 0; }
    int tch_read_stream_seek_start(void* ptr) { return 0; }
    int tch_read_stream_seek_end(void* ptr) { return 0; }
    int tch_read_stream_stream_position(void* ptr) { return 0; }
}