#ifndef INCLUDE_MACROS_H_
#define INCLUDE_MACROS_H_

#include <NvInfer.h>

#ifdef API_EXPORTS
#if defined(_MSC_VER)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif
#else

#if defined(_MSC_VER)
#define API __declspec(dllimport)
#else
#define API
#endif
#endif // API_EXPORTS

#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept
#define TRT_CONST_ENQUEUE const
#else
#define TRT_NOEXCEPT
#define TRT_CONST_ENQUEUE
#endif

#endif // INCLUDE_MACROS_H_
