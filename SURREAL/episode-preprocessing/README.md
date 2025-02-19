# SURREAL Episode Processing Pipeline

## Workflow

```mermaid
graph TD;
    A[Raw GPS Data\n*.csv] --> B[preprocess-geospatial.py]
    A --> C[Raw App Data\n*.csv]
    B --> D[{pid}_qstarz_preprocessed.csv]
    C --> E[{pid}_app_preprocessed.csv]
    D --> F[detect_episodes.py]
    E --> F
    F --> G[{pid}_episodes.csv]
    G --> H[Map-episodes.py]
    H --> I[{pid}_map.html]
```


## Processing Steps
1. **Preprocessing** (`preprocess-geospatial.py`)
   - Merges GPS/app data
   - Cleans timestamps
   - Output: `{pid}_qstarz_preprocessed.csv`, `{pid}_app_preprocessed.csv`

2. **Episode Detection** (`detect_episodes.py`)
   - Identifies movement/digital episodes
   - Output: `{pid}_episodes.csv`

3. **Visualization** (`Map-episodes.py`)
   - Generates interactive maps
   - Output: `{pid}_map.html`