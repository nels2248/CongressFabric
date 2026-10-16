# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "39418109-5bb5-4630-a82f-fc5836ac8a08",
# META       "default_lakehouse_name": "Lakehouse1",
# META       "default_lakehouse_workspace_id": "3961b5ec-d383-4029-b4d8-9c4b4acd3d4c",
# META       "known_lakehouses": [
# META         {
# META           "id": "39418109-5bb5-4630-a82f-fc5836ac8a08"
# META         }
# META       ]
# META     },
# META     "warehouse": {
# META       "default_warehouse": "e65dd74a-7993-40d8-9680-d11a6cc54539",
# META       "known_warehouses": [
# META         {
# META           "id": "e65dd74a-7993-40d8-9680-d11a6cc54539",
# META           "type": "Lakewarehouse"
# META         }
# META       ]
# META     }
# META   }
# META }

# CELL ********************

# ════════════════════════════════════════════════════════════
# CELL — Bill Text Clustering → Lakehouse
# TF-IDF + KMeans clustering of congressional bills
# Writes to congress_gold.bill_clusters
# ════════════════════════════════════════════════════════════

from pyspark.sql import functions as F
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline
from pyspark.sql.window import Window

# ── PATHS ────────────────────────────────────────────────────
LAKEHOUSE_PATH = "abfss://3961b5ec-d383-4029-b4d8-9c4b4acd3d4c@onelake.dfs.fabric.microsoft.com/39418109-5bb5-4630-a82f-fc5836ac8a08"

cluster_path = f"{LAKEHOUSE_PATH}/Tables/congress_gold/bill_clusters"
summary_path = f"{LAKEHOUSE_PATH}/Tables/congress_gold/bill_cluster_summary"

K = 20

# ── 1. LOAD DATA ─────────────────────────────────────────────
print("Loading data...")

summaries = (
    spark.table("congress_silver.summaries")
    .filter(F.col("summary_text").isNotNull() & (F.col("summary_text") != ""))
    .groupBy("bill_id")
    .agg(F.concat_ws(" ", F.collect_list("summary_text")).alias("text"))
)

bills = (
    spark.table("congress_gold.bills_master")
    .select(
        "bill_id", "bill_type", "policy_area", "sponsor_party",
        "sponsor_state", "origin_chamber", "title",
        "introduced_date", "cosponsor_count", "latest_action_text"
    )
)

df = (
    bills.join(summaries, "bill_id", "left")
    .withColumn(
        "text",
        F.when(
            F.col("text").isNotNull() & (F.col("text") != ""),
            F.concat_ws(" ", F.col("title"), F.col("text"))
        ).otherwise(F.col("title"))
    )
    .dropna(subset=["text"])
    .filter(F.col("text") != "")
    .cache()
)

print(f"Bills loaded (all bills_master rows): {df.count():,}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── 2. TF-IDF ────────────────────────────────────────────────
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, Normalizer
from pyspark.ml import Pipeline

extra_stopwords = [
    "act","section","congress","united","states","federal",
    "shall","require","provide","including","pursuant","amendment",
    "bill","amend","subtitle","title","sec","subsection","paragraph",
    "accordance","established","authorized","appropriated"
]

tokenizer = Tokenizer(inputCol="text", outputCol="words")

remover = StopWordsRemover(
    inputCol="words",
    outputCol="filtered",
    stopWords=StopWordsRemover.loadDefaultStopWords("english") + extra_stopwords
)

hashing_tf = HashingTF(inputCol="filtered", outputCol="raw_tf", numFeatures=8192)
idf = IDF(inputCol="raw_tf", outputCol="features_raw", minDocFreq=3)

# L2-normalize so document length doesn't dominate distance calcs
normalizer = Normalizer(inputCol="features_raw", outputCol="features", p=2.0)

pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf, normalizer])

print("Fitting TF-IDF...")
vectorized = pipeline.fit(df).transform(df).cache()



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#CELL IS NEEDED TO REMOVED ROWS THAT DON'T HAVE TITLES OR TEXT
from pyspark.ml.linalg import Vector
from pyspark.sql.types import BooleanType

is_nonzero = F.udf(lambda v: v.numNonzeros() > 0, BooleanType())

vectorized = vectorized.filter(is_nonzero(F.col("features")))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── 3. FINAL KMEANS ──────────────────────────────────────────
import mlflow

#HAD ELBOW CHECKING INCORORATED BEFORE BUT TOOK THIS OUT TO SIMPLIFY CODE.  COULD BE ADDED BACK IN LATER.  THIS IS A TECHNIQUE THAT IS USED TO DETERMINE THE BEST AMOUNT OF GROUPS
print(f"\nTraining final model K={K}...")

try:
    with mlflow.start_run(run_name=f"final_kmeans_k{K}"):
        kmeans = KMeans(
            featuresCol="features",
            predictionCol="cluster",
            k=K,
            seed=42,
            maxIter=50,
            distanceMeasure="cosine"
        )

        model = kmeans.fit(vectorized)
        clustered = model.transform(vectorized)

        mlflow.log_param("k", K)
        mlflow.log_param("maxIter", 50)
        mlflow.log_metric("training_cost", model.summary.trainingCost)
finally:
    mlflow.end_run()


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── 4. OUTPUT TABLE ──────────────────────────────────────────
output = clustered.select(
    "bill_id",
    "title",
    "bill_type",
    "origin_chamber",
    "policy_area",
    "sponsor_party",
    "sponsor_state",
    "introduced_date",
    "cosponsor_count",
    "latest_action_text",
    "cluster",
    F.length("text").alias("summary_char_length"),
    F.size(F.split(F.col("text"), " ")).alias("summary_word_count")
)



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── 5. WRITE CLUSTERS (CLEAN REBUILD) ───────────────────────
print("\nWriting bill_clusters...")

spark.sql("DROP TABLE IF EXISTS congress_gold.bill_clusters")

output.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(cluster_path)

spark.sql(f"""
CREATE TABLE congress_gold.bill_clusters
USING DELTA
LOCATION '{cluster_path}'
""")



# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# ── 8. CLUSTER SUMMARY ──────────────────────────────────────
print("Writing cluster summary...")

spark.sql("DROP TABLE IF EXISTS congress_gold.bill_cluster_summary")

cluster_summary = clustered.groupBy("cluster").agg(
    F.count("*").alias("bill_count"),
    F.countDistinct("policy_area").alias("distinct_policy_areas"),
    F.countDistinct("sponsor_party").alias("distinct_parties"),
    F.collect_set("policy_area").alias("policy_areas"),
    F.avg("cosponsor_count").alias("avg_cosponsors"),
    F.min("introduced_date").alias("earliest_intro"),
    F.max("introduced_date").alias("latest_intro")
)

cluster_summary.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("congress_gold.bill_cluster_summary")

print("  Registered: congress_gold.bill_cluster_summary")

# ── 9. FINAL OUTPUT ─────────────────────────────────────────
print("\n==============================")
print("CLUSTERING COMPLETE")
print("==============================")

print(f"Bills clustered: {output.count():,}")
print(f"Clusters: {output.select('cluster').distinct().count()}")

print("\nDone ✓")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#FINAL CLEANUP

from pyspark.sql import functions as F
from pyspark.sql.window import Window

print("Building CLEAN cluster labels (filtered vocabulary)...")

# ─────────────────────────────────────────────
# 1. BASE
# ─────────────────────────────────────────────
base = clustered.select("cluster", "text")

# ─────────────────────────────────────────────
# 2. VERY AGGRESSIVE STOPWORDS
# ─────────────────────────────────────────────
extra_stopwords = set([
    # generic english junk
    "this","that","these","those","with","from","into","upon","under",
    "shall","must","will","may","can","could","would","should",
    "also","such","than","then","there","here","where","what","when","other",

    # legislative boilerplate
    "section","subsection","paragraph","title","act","bill","amendment",
    "congress","federal","states","united","state","provide","require",
    "including","pursuant","accordance","authorized","established",
    "appropriated","respectively","thereof","therein","nbsp","certain",
     "program", "requires","department","other","resolution", "specifically", "requirements", "services", "national",
     "service","related","provides","information"

])

# ─────────────────────────────────────────────
# 3. TOKENIZE
# ─────────────────────────────────────────────
tokens = base.select(
    "cluster",
    F.explode(
        F.split(F.lower(F.col("text")), r"[^a-zA-Z]+")
    ).alias("token")
)

# ─────────────────────────────────────────────
# 4. CLEAN TOKENS
# ─────────────────────────────────────────────
tokens = tokens.filter(
    (F.length("token") >= 4) &                 # drop short words
    (~F.col("token").isin(list(extra_stopwords))) &
    (~F.col("token").rlike("^[0-9]+$"))       # drop numbers
)

# ─────────────────────────────────────────────
# 5. FREQUENCY FILTER (removes noise words)
# ─────────────────────────────────────────────
counts = tokens.groupBy("cluster", "token").count()

w = Window.partitionBy("cluster").orderBy(F.desc("count"))

top_words = counts.withColumn(
    "rank",
    F.row_number().over(w)
).filter(F.col("rank") <= 6)

# ─────────────────────────────────────────────
# 6. BUILD CLEAN LABELS
# ─────────────────────────────────────────────
labels = top_words.groupBy("cluster") \
    .agg(F.collect_list("token").alias("words")) \
    .withColumn(
        "cluster_label",
        F.concat_ws(", ", F.slice("words", 1, 4))
    )

# ─────────────────────────────────────────────
# 7. SIMPLE POLICY DOMAIN MAP
# ─────────────────────────────────────────────
labels = labels.withColumn(
    "policy_domain",
    F.when(F.col("cluster_label").rlike("health|medic|hospital|drug|insurance"), "Healthcare")
     .when(F.col("cluster_label").rlike("defense|military|veteran|security"), "Defense")
     .when(F.col("cluster_label").rlike("tax|budget|revenue|irs"), "Tax & Budget")
     .when(F.col("cluster_label").rlike("energy|oil|gas|power"), "Energy")
     .when(F.col("cluster_label").rlike("agriculture|farm|food"), "Agriculture")
     .when(F.col("cluster_label").rlike("education|school|student"), "Education")
     .when(F.col("cluster_label").rlike("immigration|border|visa"), "Immigration")
     .otherwise("General / Mixed")
)

# ─────────────────────────────────────────────
# 8. JOIN BACK
# ─────────────────────────────────────────────
final = clustered.select(
    "bill_id","cluster","title","bill_type","origin_chamber",
    "policy_area","sponsor_party","sponsor_state",
    "introduced_date","cosponsor_count","latest_action_text"
).join(labels, "cluster", "left")

# ─────────────────────────────────────────────
# 9. WRITE POWER BI SAFE TABLE
# ─────────────────────────────────────────────
output_path = f"{LAKEHOUSE_PATH}/Tables/congress_gold/bill_cluster_labels"

spark.sql("DROP TABLE IF EXISTS congress_gold.bill_cluster_labels")

final_pbi = final.drop("raw_tf", "features")   # strip the vector UDT columns

final_pbi.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(output_path)

spark.sql(f"""
CREATE TABLE congress_gold.bill_cluster_labels
USING DELTA
LOCATION '{output_path}'
""")

print("DONE ✓ Clean labels written for Power BI")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
