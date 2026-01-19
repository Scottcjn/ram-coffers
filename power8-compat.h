/*
 * power8-compat.h - POWER8 vec_xl/vec_xst compatibility
 *
 * DO NOT define __POWER9_VECTOR__ - that enables GCC to use POWER9 builtins!
 * Instead, provide just the vec_xl/vec_xst macros for POWER8 use.
 */
#ifndef POWER8_COMPAT_H
#define POWER8_COMPAT_H

#if defined(__POWER8_VECTOR__) && !defined(__POWER9_VECTOR__)

#include <altivec.h>
#include <string.h>

/* vec_xl for various types - use vec_ld (requires alignment) */
#ifndef vec_xl
#define vec_xl(offset, ptr) vec_ld(offset, ptr)
#endif

/* vec_xst for various types - use vec_st (requires alignment) */
#ifndef vec_xst
#define vec_xst(v, offset, ptr) vec_st(v, offset, ptr)
#endif

/* vec_xl_len - partial vector load
 * On POWER8, use memcpy into aligned buffer then load
 * This is a statement expression macro for type safety
 */
#ifndef vec_xl_len
#define vec_xl_len(ptr, len) \
    __extension__ ({ \
        union { unsigned char buf[16]; __typeof__(vec_ld(0, (ptr))) v; } __u; \
        __builtin_memset(__u.buf, 0, 16); \
        __builtin_memcpy(__u.buf, (ptr), (len) > 16 ? 16 : (len)); \
        __u.v; \
    })
#endif

/* Marker that POWER8 compat is active */
#define GGML_POWER8_COMPAT_ACTIVE 1

#endif /* __POWER8_VECTOR__ && !__POWER9_VECTOR__ */

#endif /* POWER8_COMPAT_H */
