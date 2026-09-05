| treatment   | control                 | metric   |    n |    mean_diff |           t_p |    wilcoxon_p |      cohen_d |   cliffs_delta |      t_p_holm |   wilcoxon_p_holm | subset             |
|:------------|:------------------------|:---------|-----:|-------------:|--------------:|--------------:|-------------:|---------------:|--------------:|------------------:|:-------------------|
| AIPA (full) | AIPA (rule policy)      | Hit@10   | 1077 |  0.00371402  |   0.45787     |   0.773147    |   0.0226289  |    0.000896607 |   1           |       1           | natural            |
| AIPA (full) | AIPA w/o clarification  | Hit@10   | 1077 |  0           |   1           |   0.913233    |   0          |   -0.00611072  |   1           |       1           | natural            |
| AIPA (full) | AIPA w/o counterfactual | Hit@10   | 1077 | -0.00510678  |   0.296666    |   0.436807    |  -0.0318157  |   -0.0188012   |   1           |       1           | natural            |
| AIPA (full) | AIPA w/o persistence    | Hit@10   | 1077 |  0           | nan           | nan           | nan          |  nan           | nan           |     nan           | natural            |
| AIPA (full) | AIPA w/o relationship   | Hit@10   | 1077 | -0.0116063   |   0.0378227   |   0.186328    |  -0.0633603  |   -0.0180787   |   0.302581    |       1           | natural            |
| AIPA (full) | Adaptive fusion         | Hit@10   | 1077 | -0.00278552  |   0.5447      |   0.694372    |  -0.018463   |   -0.00296397  |   1           |       1           | natural            |
| AIPA (full) | Conversation-aware      | Hit@10   | 1077 |  0.0125348   |   0.0423592   |   0.451892    |   0.0619298  |    0.0144836   |   0.302581    |       1           | natural            |
| AIPA (full) | LTP-only                | Hit@10   | 1077 |  0.0371402   |   5.03825e-09 |   0.00032345  |   0.179599   |    0.0502134   |   5.54207e-08 |       0.00355795  | natural            |
| AIPA (full) | Naive fusion            | Hit@10   | 1077 |  0.00185701  |   0.719617    |   0.966097    |   0.0109412  |   -0.00184322  |   1           |       1           | natural            |
| AIPA (full) | STI-only                | Hit@10   | 1077 | -0.0120706   |   0.027925    |   0.165305    |  -0.0670773  |   -0.0158992   |   0.251325    |       1           | natural            |
| AIPA (full) | Sequential (GRU)        | Hit@10   | 1077 |  0.0250696   |   0.000109791 |   0.0443688   |   0.118298   |    0.0282035   |   0.00109791  |       0.443688    | natural            |
| AIPA (full) | AIPA (rule policy)      | NDCG@10  | 1077 |  0.00478181  |   0.0881404   |   0.27936     |   0.0520098  |    0.000960404 |   0.705123    |       1           | natural            |
| AIPA (full) | AIPA w/o clarification  | NDCG@10  | 1077 |  0.00236752  |   0.365723    |   0.494715    |   0.0275735  |   -0.00607192  |   1           |       1           | natural            |
| AIPA (full) | AIPA w/o counterfactual | NDCG@10  | 1077 | -0.000554356 |   0.828661    |   0.3783      |  -0.00659621 |   -0.0191779   |   1           |       1           | natural            |
| AIPA (full) | AIPA w/o persistence    | NDCG@10  | 1077 |  6.91198e-05 |   0.199211    |   0.855397    |   0.0391434  |    3.36227e-05 |   1           |       1           | natural            |
| AIPA (full) | AIPA w/o relationship   | NDCG@10  | 1077 | -0.00504211  |   0.0758838   |   0.115957    |  -0.0541415  |   -0.0178752   |   0.682954    |       1           | natural            |
| AIPA (full) | Adaptive fusion         | NDCG@10  | 1077 |  0.000281586 |   0.912037    |   0.817469    |   0.00336697 |   -0.00235877  |   1           |       1           | natural            |
| AIPA (full) | Conversation-aware      | NDCG@10  | 1077 |  0.00606955  |   0.109675    |   0.636158    |   0.0487841  |    0.0138293   |   0.767725    |       1           | natural            |
| AIPA (full) | LTP-only                | NDCG@10  | 1077 |  0.0217875   |   1.97358e-08 |   2.6545e-05  |   0.172377   |    0.0502548   |   2.17094e-07 |       0.000291995 | natural            |
| AIPA (full) | Naive fusion            | NDCG@10  | 1077 |  0.00355435  |   0.180044    |   0.839017    |   0.0408771  |   -0.00155785  |   1           |       1           | natural            |
| AIPA (full) | STI-only                | NDCG@10  | 1077 | -0.00222576  |   0.44579     |   0.552619    |  -0.0232415  |   -0.0146604   |   1           |       1           | natural            |
| AIPA (full) | Sequential (GRU)        | NDCG@10  | 1077 |  0.0141143   |   0.000235961 |   0.0316967   |   0.112423   |    0.0272956   |   0.00235961  |       0.316967    | natural            |
| AIPA (full) | AIPA (rule policy)      | Hit@10   |   62 | -0.016129    |   0.418655    |   0.806788    |  -0.103413   |   -0.0169095   |   1           |       1           | conflict_natural   |
| AIPA (full) | AIPA w/o clarification  | Hit@10   |   62 | -0.0483871   |   0.0571132   |   0.342961    |  -0.246271   |   -0.0650364   |   0.456906    |       1           | conflict_natural   |
| AIPA (full) | AIPA w/o counterfactual | Hit@10   |   62 | -0.0483871   |   0.0327456   |   0.242701    |  -0.277486   |   -0.0793444   |   0.327456    |       1           | conflict_natural   |
| AIPA (full) | AIPA w/o persistence    | Hit@10   |   62 |  0           | nan           | nan           | nan          |  nan           | nan           |     nan           | conflict_natural   |
| AIPA (full) | AIPA w/o relationship   | Hit@10   |   62 | -0.0564516   |   0.0336732   |   0.177791    |  -0.275972   |   -0.0949532   |   0.327456    |       1           | conflict_natural   |
| AIPA (full) | Adaptive fusion         | Hit@10   |   62 |  0.00806452  |   0.56793     |   0.809842    |   0.0729261  |    0.0156087   |   1           |       1           | conflict_natural   |
| AIPA (full) | Conversation-aware      | Hit@10   |   62 | -0.016129    |   0.418655    |   0.645032    |  -0.103413   |   -0.0312175   |   1           |       1           | conflict_natural   |
| AIPA (full) | LTP-only                | Hit@10   |   62 |  0.0403226   |   0.0581389   |   0.342957    |   0.245232   |    0.0494277   |   0.456906    |       1           | conflict_natural   |
| AIPA (full) | Naive fusion            | Hit@10   |   62 | -0.00806452  |   0.56793     |   0.809842    |  -0.0729261  |   -0.0156087   |   1           |       1           | conflict_natural   |
| AIPA (full) | STI-only                | Hit@10   |   62 | -0.0564516   |   0.0183633   |   0.166988    |  -0.307757   |   -0.0949532   |   0.201996    |       1           | conflict_natural   |
| AIPA (full) | Sequential (GRU)        | Hit@10   |   62 |  0.0241935   |   0.369979    |   0.645944    |   0.114705   |    0.0182102   |   1           |       1           | conflict_natural   |
| AIPA (full) | AIPA (rule policy)      | NDCG@10  |   62 | -0.00650007  |   0.379443    |   0.987494    |  -0.11244    |   -0.0163892   |   1           |       1           | conflict_natural   |
| AIPA (full) | AIPA w/o clarification  | NDCG@10  |   62 | -0.0197846   |   0.0631635   |   0.464734    |  -0.240356   |   -0.0632154   |   0.694798    |       1           | conflict_natural   |
| AIPA (full) | AIPA w/o counterfactual | NDCG@10  |   62 | -0.0187855   |   0.101731    |   0.269493    |  -0.211019   |   -0.0783039   |   1           |       1           | conflict_natural   |
| AIPA (full) | AIPA w/o persistence    | NDCG@10  |   62 |  0           | nan           | nan           | nan          |  nan           | nan           |     nan           | conflict_natural   |
| AIPA (full) | AIPA w/o relationship   | NDCG@10  |   62 | -0.0185318   |   0.182971    |   0.145511    |  -0.171066   |   -0.0933923   |   1           |       1           | conflict_natural   |
| AIPA (full) | Adaptive fusion         | NDCG@10  |   62 |  0.0139314   |   0.177417    |   0.483346    |   0.173294   |    0.0169095   |   1           |       1           | conflict_natural   |
| AIPA (full) | Conversation-aware      | NDCG@10  |   62 | -0.00693983  |   0.627003    |   0.662587    |  -0.0620295  |   -0.0309573   |   1           |       1           | conflict_natural   |
| AIPA (full) | LTP-only                | NDCG@10  |   62 |  0.0201094   |   0.176685    |   0.483346    |   0.173592   |    0.0478668   |   1           |       1           | conflict_natural   |
| AIPA (full) | Naive fusion            | NDCG@10  |   62 | -0.00241731  |   0.752756    |   0.817381    |  -0.0401867  |   -0.0158689   |   1           |       1           | conflict_natural   |
| AIPA (full) | STI-only                | NDCG@10  |   62 | -0.0123407   |   0.246339    |   0.272493    |  -0.148658   |   -0.0902706   |   1           |       1           | conflict_natural   |
| AIPA (full) | Sequential (GRU)        | NDCG@10  |   62 |  0.0159791   |   0.388749    |   0.64601     |   0.110247   |    0.0176899   |   1           |       1           | conflict_natural   |
| AIPA (full) | AIPA (rule policy)      | Hit@10   |   78 |  0.0384615   |   0.20281     |   0.353019    |   0.145444   |    0.0307364   |   1           |       1           | conflict_synthetic |
| AIPA (full) | AIPA w/o clarification  | Hit@10   |   78 |  0.00641026  |   0.854074    |   0.856868    |   0.0208953  |   -0.0101907   |   1           |       1           | conflict_synthetic |
| AIPA (full) | AIPA w/o counterfactual | Hit@10   |   78 | -0.00641026  |   0.828875    |   0.602546    |  -0.0245569  |   -0.0251479   |   1           |       1           | conflict_synthetic |
| AIPA (full) | AIPA w/o persistence    | Hit@10   |   78 |  0           | nan           | nan           | nan          |  nan           | nan           |     nan           | conflict_synthetic |
| AIPA (full) | AIPA w/o relationship   | Hit@10   |   78 |  0.025641    |   0.320443    |   0.552213    |   0.113228   |    0.0299145   |   1           |       1           | conflict_synthetic |
| AIPA (full) | Adaptive fusion         | Hit@10   |   78 |  0.0192308   |   0.470409    |   0.689645    |   0.0821347  |    0.0047666   |   1           |       1           | conflict_synthetic |
| AIPA (full) | Conversation-aware      | Hit@10   |   78 |  0.185897    |   4.695e-06   |   0.000182167 |   0.557727   |    0.255753    |   4.2255e-05  |       0.0016395   | conflict_synthetic |
| AIPA (full) | LTP-only                | Hit@10   |   78 |  0.185897    |   2.60475e-06 |   0.000103421 |   0.574778   |    0.255753    |   2.60475e-05 |       0.00103421  | conflict_synthetic |
| AIPA (full) | Naive fusion            | Hit@10   |   78 |  0.0384615   |   0.20281     |   0.433906    |   0.145444   |    0.066075    |   1           |       1           | conflict_synthetic |
| AIPA (full) | STI-only                | Hit@10   |   78 |  0           |   1           |   1           |   0          |   -0.0141354   |   1           |       1           | conflict_synthetic |
| AIPA (full) | Sequential (GRU)        | Hit@10   |   78 |  0.192308    |   1.40737e-06 |   5.84234e-05 |   0.592362   |    0.259698    |   1.54811e-05 |       0.000642657 | conflict_synthetic |