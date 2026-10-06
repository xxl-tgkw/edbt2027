# Native dynamic-index run (reproducibility record)

This directory contains two independent native runs.

* **FreshDiskANN/DiskANN**: Microsoft DiskANN C++ commit
  `78256bbab4685e1774e78d331e081a153be26823`, dynamic-memory API, 800k base
  vectors and five 40k insert rounds (1M final vectors), 128 queries, and
  budgets 64/128/256/512.  `freshdiskann_*.log` are unmodified stdout logs;
  the JSON files are parsed directly from those logs.
* **SPFresh**: Microsoft SPFresh commit
  `5893eb61ee3b18610b6b00f1939be7dae1af8904`, official SPANN/SPFresh build and
  update path, 5,000 base + 1,000 inserted vectors, 100 dimensions, 32
  queries, one update round.  It was run with
  `SPFRESH_SPDK_USE_MEM_IMPL=1` because this host has no NVMe controller.
  `spfresh_memory.log` is the complete native log and reports Recall@10,
  update throughput, maintenance counts, and latency distributions.

The SPFresh memory fallback is not an SSD experiment.  In particular, the
reported SPDK latency is an in-memory backend statistic and must not be
compared with NVMe latency.  The host exposes only `/dev/vda2`; no `/dev/nvme*`
device was present, so no SPDK PCI binding, formatting, or destructive block
device operation was attempted.

The source runner is `scripts/spfresh_memory_runner.cpp`.  The official source
needed two portability changes on this host: RocksDB 6.11 headers do not
provide the blob-file options used by SPFresh (those assignments are guarded
for RocksDB >=7), and the memory fallback's final short page was fixed to copy
only the remaining bytes and zero-pad the page.  The latter fixes an
address-sanitizer-confirmed out-of-bounds read in the upstream fallback; it
does not change the SSD code path.
