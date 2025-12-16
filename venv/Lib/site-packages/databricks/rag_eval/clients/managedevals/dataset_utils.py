import json
from datetime import datetime
from typing import Union

import pandas as pd
from pyspark.sql import types as T

from databricks.rag_eval import schemas
from databricks.rag_eval.datasets import rest_entities
from databricks.rag_eval.evaluation.entities import EvalItem

_DATASET_SCHEMA = T.StructType(
    [
        T.StructField("dataset_record_id", T.StringType()),
        T.StructField("create_time", T.TimestampType()),
        T.StructField("created_by", T.StringType()),
        T.StructField("last_update_time", T.TimestampType()),
        T.StructField("last_updated_by", T.StringType()),
        T.StructField(
            "source",
            T.StructType(
                [
                    T.StructField(
                        "human",
                        T.StructType(
                            [
                                T.StructField("user_name", T.StringType()),
                            ]
                        ),
                    ),
                    T.StructField(
                        "document",
                        T.StructType(
                            [
                                T.StructField("doc_uri", T.StringType()),
                                T.StructField("content", T.StringType()),
                            ]
                        ),
                    ),
                    T.StructField(
                        "trace",
                        T.StructType(
                            [
                                T.StructField("trace_id", T.StringType()),
                            ]
                        ),
                    ),
                ]
            ),
        ),
        T.StructField("inputs", T.StringType()),  # Json serialized dict[str, Any]
        T.StructField("expectations", T.StringType()),  # Json serialized dict[str, Any]
        T.StructField("tags", T.MapType(T.StringType(), T.StringType())),
    ]
)


def _parse_timestamp(timestamp_str: Union[str, datetime]) -> datetime:
    """Parse timestamp with flexible format handling."""
    if isinstance(timestamp_str, datetime):
        return timestamp_str

    formats_to_try = [
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    for fmt in formats_to_try:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue

    raise ValueError(f"time data '{timestamp_str}' does not match any supported format: {formats_to_try}")


def eval_item_to_rest_dataset_record(
    eval_item: EvalItem,
) -> rest_entities.DatasetRecord:
    # Convert the inputs.
    if isinstance(eval_item.raw_request, dict):
        inputs = [rest_entities.Input(key=k, value=v) for k, v in eval_item.raw_request.items()]
    else:
        inputs = [rest_entities.Input(key="request", value=eval_item.raw_request)]

    # Extract any source information.
    source = None
    if eval_item.source_id and eval_item.source_type:
        source_type = eval_item.source_type.lower()
        if source_type == "synthetic_from_doc":
            source = rest_entities.Source(document=rest_entities.Document(doc_uri=eval_item.source_id))
            if eval_item.ground_truth_retrieval_context:
                # Add the retrieved context to the source document content.
                expected_retrieved_context = eval_item.ground_truth_retrieval_context.to_output_dict()
                if expected_retrieved_context:
                    source.document.content = expected_retrieved_context[0].get(schemas.CHUNK_CONTENT_COL)
        elif source_type == "document":
            source = rest_entities.Source(document=rest_entities.Document(doc_uri=eval_item.source_id))
        elif source_type == "human":
            source = rest_entities.Source(human=rest_entities.Human(user_name=eval_item.source_id))
        elif source_type == "trace":
            source = rest_entities.Source(trace=rest_entities.Trace(trace_id=eval_item.source_id))
    elif eval_item.trace:
        source = rest_entities.Source(trace=rest_entities.Trace(trace_id=eval_item.trace.info.trace_id))

    # Convert the expectations.
    expectations = {}
    if eval_item.expected_facts:
        expectations[schemas.EXPECTED_FACTS_COL] = rest_entities.ExpectationValue(value=eval_item.expected_facts)

    if eval_item.ground_truth_answer:
        expectations[schemas.EXPECTED_RESPONSE_COL] = rest_entities.ExpectationValue(
            value=eval_item.ground_truth_answer
        )

    if eval_item.guidelines:
        expectations[schemas.GUIDELINES_COL] = rest_entities.ExpectationValue(value=eval_item.guidelines)
    elif eval_item.named_guidelines:
        expectations[schemas.GUIDELINES_COL] = rest_entities.ExpectationValue(value=eval_item.named_guidelines)

    if eval_item.ground_truth_retrieval_context:
        expectations[schemas.EXPECTED_RETRIEVED_CONTEXT_COL] = rest_entities.ExpectationValue(
            value=eval_item.ground_truth_retrieval_context.to_output_dict()
        )

    if isinstance(eval_item.custom_expected, dict):
        expectations.update({k: rest_entities.ExpectationValue(value=v) for k, v in eval_item.custom_expected.items()})
    elif eval_item.custom_expected is not None:
        expectations[schemas.CUSTOM_EXPECTED_COL] = rest_entities.ExpectationValue(value=eval_item.custom_expected)

    return rest_entities.DatasetRecord(
        inputs=inputs,
        source=source,
        expectations=expectations or None,
        tags=eval_item.tags,
    )


def sync_dataset_to_uc(uc_table_name: str, df: pd.DataFrame) -> None:
    # Re-order the columns to match the spark schema order.
    df = df.reindex(columns=_DATASET_SCHEMA.fieldNames())

    # Convert the "inputs" and "expectations" to json strings.
    df["inputs"] = df["inputs"].apply(json.dumps)
    df["expectations"] = df["expectations"].apply(json.dumps)

    # Convert the timestamp columns from string to datetime.
    df["create_time"] = df["create_time"].apply(_parse_timestamp)
    df["last_update_time"] = df["last_update_time"].apply(_parse_timestamp)

    spark = _get_spark_session()
    records_df = spark.createDataFrame(df, schema=_DATASET_SCHEMA)
    records_df.write.mode("overwrite").saveAsTable(uc_table_name)


def _get_spark_session():
    try:
        from pyspark.sql import SparkSession

        return SparkSession.builder.getOrCreate()
    except Exception:
        from databricks.connect import DatabricksSession

        return DatabricksSession.builder.serverless(True).getOrCreate()
