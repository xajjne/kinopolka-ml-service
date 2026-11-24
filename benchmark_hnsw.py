"""Compare exact search and forced HNSW plans on the same PostgreSQL snapshot."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean, median
import urllib.request

from dotenv import load_dotenv
import psycopg


SERVICE_DIR = Path(__file__).resolve().parent
QUERIES = [
    "фэнтези с драконами, интригами и борьбой за власть",
    "космическое путешествие, время и отношения отца с дочерью",
    "детектив об убийстве и поиске преступника",
    "романтическая комедия о неожиданной любви",
    "хоррор о сверхъестественном в заброшенном доме",
    "анимационный фильм о дружбе и семейных отношениях",
    "мужчина страдает бессонницей и создаёт тайный бойцовский клуб",
    "историческая драма о войне и выживании",
    "супергерои объединяются для спасения мира",
    "ограбление, преступники и неожиданный поворот",
]


def embedding(query, base):
    request = urllib.request.Request(
        base.rstrip("/") + "/embed",
        data=json.dumps({"text": query}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def contains_index(plan, name):
    return plan.get("Index Name") == name or any(
        contains_index(child, name) for child in plan.get("Plans", [])
    )


def plan_nodes(plan):
    return [(plan.get("Node Type"), plan.get("Index Name"))] + [
        node for child in plan.get("Plans", []) for node in plan_nodes(child)
    ]


def measure(connection, sql, parameters, mode, ef_search, repeats):
    with connection.transaction():
        # Следующий вызов явно устанавливает режим планировщика заново.
        connection.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
        connection.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
        if mode == "exact":
            connection.execute("SET LOCAL enable_sort = on")
            connection.execute("SET LOCAL enable_mergejoin = on")
            connection.execute("SET LOCAL enable_hashjoin = on")
            connection.execute("SET LOCAL enable_indexscan = off")
            connection.execute("SET LOCAL enable_indexonlyscan = off")
            connection.execute("SET LOCAL enable_bitmapscan = off")
            connection.execute("SET LOCAL enable_seqscan = on")
        else:
            connection.execute("SET LOCAL enable_sort = off")
            connection.execute("SET LOCAL enable_mergejoin = off")
            connection.execute("SET LOCAL enable_hashjoin = off")
            connection.execute("SET LOCAL enable_indexscan = on")
            connection.execute("SET LOCAL enable_indexonlyscan = on")
            connection.execute("SET LOCAL enable_seqscan = off")

        rows = connection.execute(sql, parameters).fetchall()
        timings = []
        plans = []
        for _ in range(repeats):
            explanation = connection.execute(
                "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, parameters,
            ).fetchone()[0][0]
            timings.append(explanation["Execution Time"])
            plans.append(explanation)
        plan = plans[-1]
        used_hnsw = contains_index(plan["Plan"], "idx_title_embeddings_hnsw_cosine")
        if (mode == "hnsw") != used_hnsw:
            raise RuntimeError(f"План {mode} не соответствует ожидаемому методу поиска: {plan_nodes(plan['Plan'])}")
        return {
            "ids": [row[0] for row in rows],
            "median_execution_ms": median(timings),
            "execution_ms": timings,
            "plan": plan,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=SERVICE_DIR.parent / "backend" / ".env")
    parser.add_argument("--ml-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=SERVICE_DIR.parent / "docs" / "benchmarks" / "hnsw.json")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 30:
        parser.error("repeats должен быть от 1 до 30")
    load_dotenv(args.env_file, override=False)
    # Кодирование выполняется до измерения SQL и не входит во время поиска.
    requests = [(query, embedding(query, args.ml_url)) for query in QUERIES]
    results = []
    with psycopg.connect(
        host=os.environ["POSTGRES_HOST"], port=int(os.environ["POSTGRES_PORT"]),
        dbname=os.environ["POSTGRES_DB"], user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"], connect_timeout=10,
    ) as connection:
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        corpus = connection.execute("SELECT count(*) FROM title_embeddings").fetchone()[0]
        versions = connection.execute("SELECT version(), (SELECT extversion FROM pg_extension WHERE extname = 'vector')").fetchone()
        first_model = requests[0][1]
        if any(data["model_id"] != first_model["model_id"] or data["revision"] != first_model["revision"] for _, data in requests):
            raise RuntimeError("Версия модели изменилась во время подготовки запросов")
        compatible = connection.execute(
            "SELECT count(*) FROM title_embeddings WHERE model_name = %s AND model_revision = %s AND text_version = 1",
            (first_model["model_id"], first_model["revision"]),
        ).fetchone()[0]
        if compatible < 10:
            raise RuntimeError("Для измерения recall@10 нужно минимум 10 совместимых эмбеддингов")
        for number, (query, data) in enumerate(requests, start=1):
            for scope in ("all", "movie", "series"):
                type_filter = "" if scope == "all" else " AND t.type = %s"
                sql = f"""
                    SELECT e.title_id, e.embedding <=> %s::vector AS distance
                    FROM title_embeddings AS e JOIN titles AS t ON t.id = e.title_id
                    WHERE e.model_name = %s AND e.model_revision = %s AND e.text_version = 1
                    {type_filter}
                    ORDER BY e.embedding <=> %s::vector LIMIT 10
                """
                vector = json.dumps(data["embedding"], allow_nan=False)
                params = [vector, data["model_id"], data["revision"]]
                if scope != "all":
                    params.append(scope)
                params.append(vector)
                exact = measure(connection, sql, params, "exact", 100, args.repeats)
                if not exact["ids"]:
                    continue
                for ef in (20, 40, 100, 200):
                    approximate = measure(connection, sql, params, "hnsw", ef, args.repeats)
                    recall = len(set(exact["ids"]) & set(approximate["ids"])) / len(exact["ids"])
                    results.append({
                        "query": query, "scope": scope, "ef_search": ef, "recall_at_10": recall,
                        "exact": exact, "hnsw": approximate,
                    })
                print(f"Запрос {number}/{len(requests)}, фильтр {scope}: измерения завершены", flush=True)

        # Обычный выбор планировщика без принуждения метода.
        connection.execute("SET LOCAL enable_seqscan = on")
        connection.execute("SET LOCAL enable_sort = on")
        connection.execute("SET LOCAL enable_indexscan = on")
        connection.execute("SET LOCAL enable_indexonlyscan = on")
        connection.execute("SET LOCAL enable_bitmapscan = on")
        connection.execute("SET LOCAL enable_mergejoin = on")
        connection.execute("SET LOCAL enable_hashjoin = on")
        connection.execute("SET LOCAL hnsw.ef_search = 100")
        automatic_plan = connection.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql, params).fetchone()[0][0]

    summary = []
    for ef in (20, 40, 100, 200):
        subset = [row for row in results if row["ef_search"] == ef]
        summary.append({
            "ef_search": ef,
            "mean_recall_at_10": mean(row["recall_at_10"] for row in subset),
            "median_exact_ms": median(row["exact"]["median_execution_ms"] for row in subset),
            "median_hnsw_ms": median(row["hnsw"]["median_execution_ms"] for row in subset),
        })
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "corpus_size": corpus,
        "compatible_embeddings": compatible, "postgres": versions[0], "pgvector": versions[1],
        "model": {key: first_model[key] for key in ("model_id", "revision", "dimensions")},
        "repeats": args.repeats, "query_count": len(requests), "summary": summary,
        "method": "Warm-cache SQL Execution Time from EXPLAIN ANALYZE. Exact index scans disabled; HNSW sequential scans and explicit sorts discouraged, hash/merge joins disabled to obtain the ANN plan. Timings exclude query embedding. Single machine, one snapshot. Recall uses exact top-10 ID overlap; ties may affect overlap. Automatic planner choice captured separately.",
        "automatic_plan_last_scenario": automatic_plan, "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"corpus_size": corpus, "summary": summary, "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
