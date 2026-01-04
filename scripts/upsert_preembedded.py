import argparse
import time
import numpy as np
import pandas as pd

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )


def get_points_count(client: QdrantClient, name: str) -> int:
    info = client.get_collection(name)
    return int(info.points_count or 0)


def upsert_with_retry(client: QdrantClient, collection: str, points, max_retries: int = 5) -> None:
    delay = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            client.upsert(collection_name=collection, points=points)
            return
        except (ResponseHandlingException, TimeoutError) as e:
            # timeouts / transport issues
            if attempt == max_retries:
                raise
            print(f"[retry] upsert timeout (attempt {attempt}/{max_retries}) -> sleep {delay:.1f}s")
            time.sleep(delay)
            delay *= 2
        except UnexpectedResponse as e:
            # propagate non-timeout 4xx/5xx
            raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vectors", required=True)
    p.add_argument("--payload", required=True)
    p.add_argument("--qdrant-url", required=True)
    p.add_argument("--collection", required=True)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--light-payload", action="store_true", help="payload only source_id (smaller)")
    p.add_argument("--with-title", action="store_true", help="payload includes title (bigger)")
    p.add_argument("--start-from", type=int, default=-1, help="override resume offset (default: auto)")
    args = p.parse_args()

    if args.light_payload and args.with_title:
        raise ValueError("Choose only one: --light-payload OR --with-title")

    data = np.load(args.vectors)
    qid = data["qid"].astype(np.int64)
    vec = data["vector"].astype(np.float32)

    payload_df = pd.read_parquet(args.payload).sort_values("qid").reset_index(drop=True)

    # Align vectors to qid order
    order = np.argsort(qid)
    qid = qid[order]
    vec = vec[order]

    if len(payload_df) != len(qid):
        raise ValueError(f"Size mismatch: payload={len(payload_df)} vectors={len(qid)}")

    if args.light_payload:
        payloads = payload_df[["source_id"]].to_dict("records")
    else:
        # default keep title+source_id (bigger)
        payloads = payload_df[["title", "source_id"]].to_dict("records")

    client = QdrantClient(url=args.qdrant_url, timeout=args.timeout)

    ensure_collection(client, args.collection, int(vec.shape[1]))

    n = len(qid)

    # resume logic
    if args.start_from >= 0:
        start = args.start_from
    else:
        start = get_points_count(client, args.collection)

    print(f"[upsert] total={n:,} | starting from {start:,} | batch={args.batch_size}")

    B = args.batch_size
    s = start

    while s < n:
        e = min(n, s + B)

        points = [
            qm.PointStruct(id=int(qid[i]), vector=vec[i].tolist(), payload=payloads[i])
            for i in range(s, e)
        ]

        try:
            upsert_with_retry(client, args.collection, points, max_retries=5)
            s = e
            print(f"upserted {s:,}/{n:,}")
        except ResponseHandlingException as ex:
            # if still timing out after retries -> reduce batch and continue
            if B > 200:
                new_B = max(200, B // 2)
                print(f"[warn] persistent timeout at {s:,}. Reducing batch {B} -> {new_B} and retrying.")
                B = new_B
            else:
                raise

    print("done")


if __name__ == "__main__":
    main()
