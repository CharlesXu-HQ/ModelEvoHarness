# Graph neighborhood and proximity components

A graph can supply useful item representations when edges and neighborhoods carry information unavailable in independent feature vectors. This family contains GraphSAGE, PinSage, SDNE and LINE tensor cores. The components consume prepared tensors; they do not crawl a graph, sample neighbors, generate walks or define a production graph store.

## Mechanisms and tensor contracts

GraphSAGE concatenates self features with the masked mean of neighboring features, projects them and normalizes the output. PinSage first transforms each neighbor, then aggregates it using supplied nonnegative visit weights. The visit weights encode the random-walk importance assumption and must not be replaced silently by arbitrary scores. An empty neighborhood produces a self-only transformation.

SDNE reconstructs supplied adjacency rows and penalizes distances on supplied weighted edges. Its nonzero-entry multiplier is inside the squared reconstruction error. This dense reference is appropriate for small graphs; it is not a scalable adjacency materializer. LINE provides first-order shared factors and second-order source/context factors, together with logistic objectives over explicit negative IDs. Negative sampling remains a separate host decision.

## Controlled experiment and failure conditions

Freeze the graph cutoff, candidate universe and evaluation split. Compare aggregation against self features only, then compare mean and importance-weighted aggregation with matching widths. For proximity objectives, preserve the same edges and negatives. Report retrieval metrics alongside low-degree and unseen-node slices, memory and latency. Reject gains that arise from future edges, changing eligible candidates or graph snapshots unavailable at inference. A supplied graph capability does not prove that its edges are causally meaningful.

## Research lineage

The source PDFs and immutable repository references are linked through each method card and the Reco-papers inventory. Implementations are independent compact tensor references; author-specific distributed graph pipelines are outside their contracts.
