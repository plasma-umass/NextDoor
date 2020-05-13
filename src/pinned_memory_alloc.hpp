#include <heaps/top/all.h>
#include <heaps/objectrep/all.h>
#include <heaps/debug/all.h>
#include <heaps/utility/all.h>
#include <heaps/general/kingsleyheap.h>
#include <wrappers/all.h>
#include "utils.hpp"

#include <set>
#include <unordered_map>
#include <vector>
#include <unordered_set>
#include <forward_list>
#include <list>

#ifndef __PINNED_MEMORY_ALLOC__
#define __PINNED_MEMORY_ALLOC__

namespace PinnedMemory {
  // namespace {
    class PrivateCudaMallocHostHeap {
    public:
    enum { Alignment = 1};
      static inline void * malloc (size_t sz) {
        void* ptr;
        // Round up to the size of a page.
        sz = (sz + HL::CPUInfo::PageSize - 1) & (size_t) ~(HL::CPUInfo::PageSize - 1);
        //CHK_CU(cudaMallocHost(&ptr, sz));
        ptr = mmap(
    NULL,   // Map from the start of the 2^20th page
    sz,                         // for one page length
    PROT_READ|PROT_WRITE,
    MAP_ANON|MAP_PRIVATE,             // to a private block of hardware memory
    0,
    0
  );
        return ptr;
      }
      
      static void free (void * ptr, size_t sz)
      {
        if ((long) sz < 0) {
          abort();
        }
        //CHK_CU(cudaFreeHost (reinterpret_cast<char *>(ptr)));
      }
    };

    class PrivateMMapHeap {
    public:
    enum { Alignment = 1};
      static inline void * malloc (size_t sz) {
        void* ptr;
        // Round up to the size of a page.
        sz = (sz + HL::CPUInfo::PageSize - 1) & (size_t) ~(HL::CPUInfo::PageSize - 1);
        //CHK_CU(cudaMallocHost(&ptr, sz));
        ptr = mmap(
    NULL,   // Map from the start of the 2^20th page
    sz,                         // for one page length
    PROT_READ|PROT_WRITE,
    MAP_ANON|MAP_PRIVATE,             // to a private block of hardware memory
    0,
    0
  );
        return ptr;
      }
      
      static void free (void * ptr, size_t sz)
      {
        if ((long) sz < 0) {
          abort();
        }
        //CHK_CU(cudaFreeHost (reinterpret_cast<char *>(ptr)));
      }
    };

    class CudaMallocHostHeap : public PrivateCudaMallocHostHeap {
    private:

      // Note: we never reclaim memory obtained for MyHeap, even when
      // this heap is destroyed.
      class MyHeap : public HL::LockedHeap<HL::PosixLockType, HL::FreelistHeap<HL::BumpAlloc<16 * 1024, PrivateMMapHeap> > > {
      };
      typedef HL::MyHashMap<void *, size_t, MyHeap> mapType;

    protected:
      mapType MyMap;

      HL::PosixLockType MyMapLock;

    public:
      enum { Alignment = 16};

      inline void * malloc (size_t sz) {
        void * ptr = PrivateCudaMallocHostHeap::malloc (sz);
        MyMapLock.lock();
        MyMap.set (ptr, sz);
        MyMapLock.unlock();
        return const_cast<void *>(ptr);
      }

      inline size_t getSize (void * ptr) {
        MyMapLock.lock();
        size_t sz = MyMap.get (ptr);
        MyMapLock.unlock();
        return sz;
      }

      inline void free (void * ptr) {
        MyMapLock.lock();
        size_t sz = MyMap.get (ptr);
        PrivateCudaMallocHostHeap::free (ptr, sz);
        MyMap.erase (ptr);
        MyMapLock.unlock();
      }
    };

    class InternalTopHeap : public HL::SizeHeap<HL::MmapHeap> {
    private:
    public:
    };
  // }

  class PinnedMemoryHeap : public HL::ExactlyOneHeap<HL::LockedHeap<HL::PosixLockType, HL::DebugHeap< HL::MmapHeap>>>
  //public HL::ExactlyOneHeap<HL::LockedHeap<HL::PosixLockType, HL::DebugHeap<HL::KingsleyHeap<HL::FreelistHeap<InternalTopHeap>, CudaMallocHostHeap>>>> 
  {
  protected:
    ///typedef HL::ExactlyOneHeap<HL::LockedHeap<HL::PosixLockType, HL::DebugHeap<HL::KingsleyHeap<HL::FreelistHeap<InternalTopHeap>, CudaMallocHostHeap>>>>
    
        //SuperHeap;

  public:
    PinnedMemoryHeap() {
      static_assert(Alignment % 16 == 0, "16-byte alignment");
    }
  };

  static PinnedMemoryHeap pinned_memory_heap;

  // template <typename K, typename V>
  // using unordered_map = std::unordered_map<K, V, std::hash<K>, std::equal_to<K>, HL::STLAllocator<pair<const K, V>, PinnedMemoryHeap>>;
  // template <typename K>
  // using unordered_set = std::unordered_set<K, std::hash<K>, std::equal_to<K>, HL::STLAllocator<K, PinnedMemoryHeap>>;
  // template <class T, class Compare=less<T>>
  // using set = std::set<T, Compare, HL::STLAllocator<T, PinnedMemoryHeap>>;
  // template <typename T>
  // using list = std::list<T, HL::STLAllocator<T, PinnedMemoryHeap>>;
  // template <typename T>
  // using forward_list = std::forward_list<T, HL::STLAllocator<T, PinnedMemoryHeap>>;
  // template <typename T>
  // using deque = std::deque<T, HL::STLAllocator<T, PinnedMemoryHeap>>;
  // template <typename T>
  // using queue = std::queue<T, deque<T>>;
  // template <typename T>
  // using vector = std::vector<T, HL::STLAllocator<T, PinnedMemoryHeap>>;
};

#endif