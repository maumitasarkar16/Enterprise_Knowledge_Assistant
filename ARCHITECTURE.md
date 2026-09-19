# Architecture — Dual-Mode Enterprise Knowledge Assistant

```
                         USER MESSAGE
                              |
                              v
                    +-------------------+
                    | Context-Aware     |
                    | Query Router      |
                    +---------+---------+
                              |
                 +------------+-------------+
                 |                          |
              GENERAL                  ENTERPRISE
                 |                          |
                 v                          v
        General OpenAI Answer      Standalone Query Rewrite
                                            |
                                            v
                                  +---------+---------+
                                  |                   |
                                  v                   v
                           Dense Retrieval           BM25
                           (ChromaDB)              Retrieval
                                  |                   |
                                  +---------+---------+
                                            v
                                   Reciprocal Rank Fusion
                                            |
                                            v
                                   Cross-Encoder Reranker
                                            |
                                            v
                                      Relevance Guard
                                      /             \
                                 weak                 strong
                                  |                     |
                                  v                     v
                         Enterprise NOT_FOUND      Context [S#]
                                                        |
                                                        v
                                               Grounded OpenAI
                                                        |
                                                        v
                                                Answer + Sources
```

## Key design rule
GENERAL questions may use normal model knowledge. ENTERPRISE questions are never answered from general model knowledge. If the local corpus does not support an enterprise fact, the system returns an explicit not-found response.

## Memory
Recent completed chat turns are passed to the router. Enterprise follow-ups are also rewritten into standalone retrieval queries. The final enterprise answer is still grounded only in retrieved local document chunks.
