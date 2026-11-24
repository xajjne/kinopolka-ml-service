SELECT t.id, t.title,
       NULLIF(btrim(t.overview), '') IS NOT NULL AS has_description,
       concat_ws(E'\n',
         'Название: ' || t.title,
         CASE WHEN NULLIF(btrim(t.original_title), '') IS NOT NULL
                   AND t.original_title <> t.title
              THEN 'Оригинальное название: ' || t.original_title END,
         'Описание: ' || NULLIF(btrim(t.overview), ''),
         'Жанры: ' || g.names,
         'Ключевые слова: ' || k.names
       ) AS source_text,
       e.source_hash, e.model_name, e.model_revision, e.text_version
FROM titles AS t
LEFT JOIN LATERAL (
  SELECT string_agg(g.name, ', ' ORDER BY g.name, g.tmdb_id) AS names
  FROM title_genres AS tg JOIN genres AS g ON g.id = tg.genre_id
  WHERE tg.title_id = t.id
) AS g ON true
LEFT JOIN LATERAL (
  SELECT string_agg(k.name, ', ' ORDER BY k.name, k.tmdb_id) AS names
  FROM title_keywords AS tk JOIN keywords AS k ON k.id = tk.keyword_id
  WHERE tk.title_id = t.id
) AS k ON true
LEFT JOIN title_embeddings AS e ON e.title_id = t.id
WHERE t.id > %s
ORDER BY t.id
LIMIT %s
