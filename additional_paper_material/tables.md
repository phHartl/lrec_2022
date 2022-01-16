## Configurations of CMTR-BERT in detail
| Model name              | Source text           | Title + Text | Source URLs | Authors | Tweet texts | Amount of retweets | Hierarchical Input Representation |
|-------------------------|-----------------------|--------------|-------------|---------|-------------|--------------------|-----------------------------------|
| BERT-baseline           | Original              | Yes          | No          | No      | No          | No                 | No                                |
| BERT                    | Original              | Yes          | No          | No      | No          | No                 | No                                |
| CMTR-BERT O             | Original              | Yes          | Yes         | Yes     | Yes         | Yes                | Yes                               |
| CMTR-BERT A             | Abstractive Summaries | Yes          | Yes         | Yes     | Yes         | Yes                | Yes                               |
| CMTR-BERT E             | Extractive Summaries  | Yes          | Yes         | Yes     | Yes         | Yes                | Yes                               |
| CMTR-BERT C             | None                  | No           | Yes         | Yes     | Yes         | Yes                | Yes                               |
| CMTR-BERT               | All (Ensemble)        | Yes          | Yes         | Yes     | Yes         | Yes                | Yes                               |
| CMTR-BERT O w/o context | Original              | Yes          | No          | No      | No          | No                 | Yes                               |
| CMTR-BERT A w/o context | Abstractive Summaries | Yes          | No          | No      | No          | No                 | Yes                               |
| CMTR-BERT E w/o context | Extractive Summaries  | Yes          | No          | No      | No          | No                 | Yes                               |
| CMTR-BERT w/o context   | All (Ensemble)        | Yes          | No          | No      | No          | NO                 | Yes                               |

## CMTR-BERT O with selected context features only
| Dataset    | Metric    | Text w/ sources | Text /w authors | Text w/ tweets | Text w/ retweets |
|------------|-----------|-----------------|-----------------|----------------|------------------|
| Politifact | Accuracy  | 0.930           | 0.928           | 0.925          | **0.950**            |
|            | Precision | **0.946**           | 0.933           | 0.928          | **0.946**            |
|            | Recall    | 0.900           | 0.908           | 0.907          | **0.945**            |
|            | F1        | 0.921           | 0.920           | 0.916          | **0.945**            |
| Gossipcop  | Accuracy  | **0.933**           | 0.865           | 0.906          | 0.865            |
|            | Precision | **0.893**           | 0.755           | 0.853          | 0.754            |
|            | Recall    | **0.821**           | 0.657           | 0.739          | 0.655            |
|            | F1        | **0.856**           | 0.702           | 0.791          | 0.701            |

## All configurations of CMTR-BERT without context information
| Dataset    | Metric    | Sources | Authors | Tweets | Retweets | All (CMTR-BERT C) |
|------------|-----------|---------|---------|--------|----------|-------------------|
| Politifact | Accuracy  | 0.795   | 0.475   | 0.650  | 0.608    | **0.912**             |
|            | Precision | 0.815   | 0.448   | 0.601  | 0.482    | **0.977**             |
|            | Recall    | 0.712   | 0.684   | 0.687  | 0.617    | **0.827**             |
|            | F1        | 0.757   | 0.534   | 0.639  | 0.501    | **0.895**             |
| Gossipcop  | Accuracy  | 0.911   | 0.501   | 0.878  | 0.448    | **0.957**             |
|            | Precision | 0.734   | 0.245   | 0.713  | 0.145    | **0.859**             |
|            | Recall    | **0.988**   | 0.512   | 0.846  | 0.600    | 0.985             |
|            | F1        | 0.842   | 0.327   | 0.772  | 0.233    | **0.918**             |

## CMTR-BERT content models trained on one domain of FakeNewsNet and applied to the other
| Dataset                | Metric    | BERT-Baseline | CMTR-BERT O w/o context | CMTR-BERT A w/o context | CMTR-BERT E w/o context | CMTR-BERT w/o context |
|------------------------|-----------|---------------|-------------------------|-------------------------|-------------------------|-----------------------|
| Politifact → GossipCop | Accuracy  | 0.309         | 0.304                   | 0.295                   | **0.311**               | 0.294                 |
|                        | Precision | **0.250**     | 0.248                   | 0.247                   | 0.248                   | 0.247                 |
|                        | Recall    | 0.931         | 0.925                   | 0.935                   | 0.912                   | **0.936**             |
|                        | F1        | **0.394**     | 0.391                   | 0.392                   | 0.390                   | 0.390                 |
| Gossipcop → Politifact | Accuracy  | **0.515**     | 0.473                   | 0.463                   | 0.502                   | 0.505                 |
|                        | Precision | **0.472**     | 0.411                   | 0.398                   | 0.442                   | 0.443                 |
|                        | Recall    | **0.564**     | 0.354                   | 0.351                   | 0.355                   | 0.333                 |
|                        | F1        | **0.514**     | 0.379                   | 0.371                   | 0.392                   | 0.380                 |

## CMTR-BERT context models trained on one domain of FakeNewsNet and applied to the other
| Dataset                | Metric    | Sources   | Authors   | Tweets    | Retweets | CMTR-BERT C |
|------------------------|-----------|-----------|-----------|-----------|----------|-------------|
| Politifact → GossipCop | Accuracy  | **0.501** | 0.391     | 0.270     | 0.448    | 0.423       |
|                        | Precision | **0.243** | 0.241     | 0.226     | 0.145    | 0.239       |
|                        | Recall    | 0.500     | 0.711     | **0.836** | 0.600    | 0.629       |
|                        | F1        | 0.326     | **0.359** | 0.356     | 0.233    | 0.345       |
| Gossipcop → Politifact | Accuracy  | 0.401     | 0.513     | **0.521** | 0.428    | 0.391       |
|                        | Precision | 0.411     | **0.508** | 0.484     | 0.349    | 0.404       |
|                        | Recall    | **0.736** | 0.457     | 0.649     | 0.583    | 0.731       |
|                        | F1        | 0.528     | 0.427     | **0.552** | 0.419    | 0.519       |