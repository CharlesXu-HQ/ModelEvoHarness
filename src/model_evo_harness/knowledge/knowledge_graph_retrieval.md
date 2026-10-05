# Typed knowledge-graph retrieval

A typed knowledge graph links items to entities and relations, such as brand, creator, category, or topic. Graph traversal or relation-aware embeddings may retrieve items missed by co-interaction neighbors. The relation type matters: sharing a creator and sharing a broad category should not be treated as identical evidence.

## Data and conditions

Require a versioned item catalog, item-entity links, typed triples, relation validity timestamps, and a query or seed-entity construction path. Entity resolution errors and high-degree hubs can dominate retrieval. Keep future edges out of historical evaluation. If the graph is only an unordered list of item tags, a simpler attribute retriever is the appropriate control.

## Controlled experiment

Compare popularity/collaborative retrieval, attribute overlap, and one graph relation or embedding method with the same catalog, top-K, temporal split, and downstream ranker. Ablate relation types and inspect graph coverage, cold items, hub bias, latency, and update cost. Report contribution beyond direct side attributes.

## Interpretation and feature needs

A gain solely from catalog metadata does not establish value from graph reasoning. Missing item-entity links or typed relations are direct prerequisites; a domain expert can identify this gap before any graph model runs. If they exist, test simpler attribute and neighbor methods first.

## Research lineage

This guide is independently written. Pinned upstream materials used for orientation: [recbole](https://recbole.io/docs/user_guide/data/atomic_files.html).
