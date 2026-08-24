-- Google ClusterData2019 v3, cell a, bounded task-summary export.
--
-- Run this query only after reviewing BigQuery's dry-run byte estimate. Supply
-- named INT64 parameters start_time_us, end_time_us, and finish_cutoff_time_us.
-- To use another cell, change the table suffix here and `cell` in the conversion
-- config together. Save the exact executed SQL beside the exported CSV.
WITH bounded_events AS (
  SELECT
    time,
    type,
    collection_id,
    instance_index,
    scheduling_class,
    missing_type,
    collection_type,
    priority,
    alloc_collection_id,
    resource_request.cpus AS requested_cpu
  FROM `google.com:google-cluster-data`.clusterdata_2019_a.instance_events
  WHERE time >= @start_time_us
    AND time < @finish_cutoff_time_us
),
task_events AS (
  SELECT
    collection_id,
    instance_index,
    ARRAY_AGG(
      IF(
        type = 0 AND time >= @start_time_us AND time < @end_time_us,
        STRUCT(
          time AS submit_time_us,
          requested_cpu AS requested_cpu,
          priority AS priority,
          scheduling_class AS scheduling_class,
          collection_type AS collection_type,
          COALESCE(alloc_collection_id, 0) AS alloc_collection_id
        ),
        NULL
      )
      IGNORE NULLS
      ORDER BY time
      LIMIT 1
    )[SAFE_OFFSET(0)] AS submit,
    MIN(IF(type = 6, time, NULL)) AS finish_time_us,
    MAX(IF(type IN (0, 6), COALESCE(missing_type, 999), 0)) AS missing_type,
    COUNTIF(type = 0 AND time >= @start_time_us AND time < @end_time_us) AS submit_count,
    COUNTIF(type = 6) AS finish_count
  FROM bounded_events
  WHERE collection_id IS NOT NULL
    AND instance_index IS NOT NULL
  GROUP BY collection_id, instance_index
)
SELECT
  collection_id,
  instance_index,
  submit.submit_time_us AS submit_time_us,
  finish_time_us,
  submit.requested_cpu AS requested_cpu,
  submit.priority AS priority,
  submit.scheduling_class AS scheduling_class,
  missing_type,
  submit.collection_type AS collection_type,
  submit.alloc_collection_id AS alloc_collection_id,
  submit_count,
  finish_count
FROM task_events
WHERE submit IS NOT NULL
ORDER BY collection_id, instance_index;
