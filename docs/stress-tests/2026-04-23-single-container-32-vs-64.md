# Single-Container Modal Stress Test: 32 vs 64 Inputs

Date: 2026-04-23
Branch: `Noel/stress-testing`
Endpoint: `https://noelalex404--bank-parser.modal.run`
GPU: A100 80GB
CPU: 12 cores
Memory: 40960 MiB
Max containers: 1
Corpus: `bank statements/`
Unique PDFs: 56
Target requests per run: 250
Request timeout: 1800 seconds

## Runs

| Config | Results Folder | Success | Duration | Throughput | Mean Latency | P50 | P90 | P95 | P99 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `max_inputs=64`, `target_inputs=64` | `load-test-results/20260423T095749Z` | 245/250 | 1254.4s | 0.199 req/s | 282.7s | 262.9s | 369.2s | 491.1s | 1090.2s | 1212.0s |
| `max_inputs=32`, `target_inputs=32` | `load-test-results/20260423T103148Z` | 246/250 | 1232.4s | 0.203 req/s | 150.0s | 148.9s | 213.1s | 236.3s | 274.6s | 350.5s |

## Server-Side Metrics

| Config | Server Mean | Server P50 | Server P95 | OCR Mean | OCR Queue Wait Mean | OCR Request Mean | Client-Server Gap Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `64` | 186.6s | 181.9s | 287.4s | 105.1s | 33.7s | 58.7s | 85.4s |
| `32` | 147.2s | 145.8s | 230.7s | 90.2s | 21.6s | 56.3s | 3.4s |

## Result

`32` concurrent inputs is the better one-container setting.

Throughput was only slightly better than `64`, but latency was much better:

- Mean latency dropped from 282.7s to 150.0s.
- P95 latency dropped from 491.1s to 236.3s.
- P99 latency dropped from 1090.2s to 274.6s.
- Max latency dropped from 1212.0s to 350.5s.
- Client-minus-server gap dropped from 85.4s to 3.4s, which means `64` was mostly creating admission/backlog delay rather than useful GPU throughput.

Recommendation: keep one-container stress testing and production admission at `max_inputs=32`, `target_inputs=32` unless a later run proves a higher setting materially improves throughput without reintroducing long-tail backlog.
