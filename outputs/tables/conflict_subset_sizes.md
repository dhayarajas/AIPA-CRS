| subset             | definition                                                       |    n |
|:-------------------|:-----------------------------------------------------------------|-----:|
| strict             | weak-rule label in Conflict/Override                             |   62 |
| broad              | Conflict/Override or (confidence >= 0.6 and JS(ltp, sti) >= 0.5) |  256 |
| broad_only         | broad minus strict                                               |  194 |
| synthetic_conflict | synthetic Conflict/Override                                      |   78 |
| natural            | all natural test instances                                       | 1077 |