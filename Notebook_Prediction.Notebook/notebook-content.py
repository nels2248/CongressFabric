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

# MARKDOWN ********************


# MARKDOWN ********************

# Cell 1 — Config

# CELL ********************

LAKEHOUSE_PATH = "abfss://3961b5ec-d383-4029-b4d8-9c4b4acd3d4c@onelake.dfs.fabric.microsoft.com/39418109-5bb5-4630-a82f-fc5836ac8a08"

print("Cell 1 loaded — config ready")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Cell 2 — Imports

# CELL ********************

from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F

print("Cell 2 loaded — libraries imported")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Cell 3 — Load and inspect features table

# CELL ********************

features_df = spark.table("congress_gold.bill_ml_features")

print(f"Row count: {features_df.count():,}")
features_df.printSchema()
display(features_df.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Cell 4 — Define the target (adjust column name if needed)

# CELL ********************

# Cell 4 — Define target + fill nulls on numeric features
model_df = features_df.withColumn(
    "target",
    F.when(F.col("bill_status_label").isin("passed_house", "passed_senate", "enacted"), 1).otherwise(0)
).fillna({"committee_count": 0, "action_count": 0})

print("Row count after target + fillna:", model_df.count())
model_df.groupBy("target").count().show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Cell 5 — Feature pipeline (adjust column lists based on Cell 3 output)

# CELL ********************

# Cell 5 — Feature pipeline 
categorical_cols = ["bill_type", "sponsor_is_republican", "policy_area", "origin_chamber"]
numeric_cols = ["committee_count", "action_count","cosponsor_count", "amendment_count", "related_bills_count", "subjects_count", "title_count"]

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec") for c in categorical_cols]

assembler_inputs = [f"{c}_vec" for c in categorical_cols] + numeric_cols
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features", handleInvalid="keep")

pipeline = Pipeline(stages=indexers + encoders + [assembler])

print("Cell 5 loaded — feature pipeline defined")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Cell 6 — Train/test split + fit pipeline

# CELL ********************

train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=42)

pipeline_model = pipeline.fit(train_df)
train_prepped = pipeline_model.transform(train_df)
test_prepped = pipeline_model.transform(test_df)

print(f"Train: {train_prepped.count():,} rows | Test: {test_prepped.count():,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************


# MARKDOWN ********************

# Cell 7 — Train Random Forest

# CELL ********************

rf = RandomForestClassifier(featuresCol="features", labelCol="target", numTrees=100, maxDepth=8, seed=42)
rf_model = rf.fit(train_prepped)

predictions = rf_model.transform(test_prepped)
print("✓ Model trained")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Cell 8 - Evaluate

# CELL ********************

auc_eval = BinaryClassificationEvaluator(labelCol="target", metricName="areaUnderROC")
acc_eval = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")

auc = auc_eval.evaluate(predictions)
accuracy = acc_eval.evaluate(predictions)

print(f"AUC-ROC:  {auc:.3f}")
print(f"Accuracy: {accuracy:.3f}")

predictions.groupBy("target", "prediction").count().show()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Cell 9 — Feature importance

# CELL ********************

importances = rf_model.featureImportances.toArray()
importance_df = spark.createDataFrame(
    [(assembler_inputs[i], float(importances[i])) for i in range(len(assembler_inputs))],
    ["feature", "importance"]
).orderBy(F.desc("importance"))

display(importance_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 9c — Refit pipeline + model on FULL dataset for production scoring
# (train/test split above is validation-only — this is the deployed model)

full_prepped = pipeline.fit(model_df).transform(model_df)

rf_final = RandomForestClassifier(
    featuresCol="features", labelCol="target",
    numTrees=100, maxDepth=8, seed=42
)
rf_final_model = rf_final.fit(full_prepped)

full_predictions = rf_final_model.transform(full_prepped)
print(f"Full-population predictions: {full_predictions.count():,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Cell 10 — Write predictions + importances to Gold

# CELL ********************

from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col

spark.sql("CREATE SCHEMA IF NOT EXISTS congress_gold")

spark.sql("DROP TABLE IF EXISTS congress_gold.bill_predictions")

# was: predictions.withColumn(...)... .saveAsTable("congress_gold.bill_predictions")
full_predictions.withColumn("prob_pass", vector_to_array(col("probability"))[1]) \
    .select("bill_id", "target", "prediction", "prob_pass") \
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("congress_gold.bill_predictions")

importance_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("congress_gold.bill_feature_importance")

print("  Registered: congress_gold.bill_predictions")
print("  Registered: congress_gold.bill_feature_importance")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Cell 11 — Validation summary

# CELL ********************

print("=" * 55)
print("PREDICTION MODEL VALIDATION SUMMARY")
print("=" * 55)
print(f"AUC-ROC:  {auc:.3f}")
print(f"Accuracy: {accuracy:.3f}")
spark.sql("SELECT target, prediction, COUNT(*) AS cnt FROM congress_gold.bill_predictions GROUP BY target, prediction").show()

print("Prediction model complete. Tables available in:")
print("  • congress_gold.bill_predictions")
print("  • congress_gold.bill_feature_importance")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier as SklearnRF

# ── Evaluation (test set only — this is your honest generalization estimate) ──
auc_eval = BinaryClassificationEvaluator(labelCol="target", metricName="areaUnderROC")
acc_eval = MulticlassClassificationEvaluator(labelCol="target", predictionCol="prediction", metricName="accuracy")

auc = auc_eval.evaluate(predictions)
accuracy = acc_eval.evaluate(predictions)

print(f"AUC-ROC:  {auc:.3f}")
print(f"Accuracy: {accuracy:.3f}")

predictions.groupBy("target", "prediction").count().show()

# ── Refit target typo fix (passsed_senate -> passed_senate) ───────
model_df = features_df.withColumn(
    "target",
    F.when(F.col("bill_status_label").isin("passed_house", "passed_senate", "enacted"), 1).otherwise(0)
).fillna({"committee_count": 0, "action_count": 0})

# ── Refit pipeline + model on FULL population for production scoring ──
# (train/test split above stays validation-only; this is the deployed model)
full_prepped = pipeline.fit(model_df).transform(model_df)

rf_final = RandomForestClassifier(
    featuresCol="features", labelCol="target",
    numTrees=rf_model.getNumTrees, maxDepth=rf_model.getOrDefault("maxDepth"), seed=42
)
rf_final_model = rf_final.fit(full_prepped)
full_predictions = rf_final_model.transform(full_prepped)

print(f"Full-population predictions: {full_predictions.count():,} rows "
      f"(vs. test-only: {predictions.count():,})")

# ── Real feature names (handles one-hot expansion correctly) ───
def get_feature_names(df, features_col="features"):
    attrs = df.schema[features_col].metadata["ml_attr"]["attrs"]
    names = []
    for attr_type in ("numeric", "binary", "nominal"):
        if attr_type in attrs:
            for a in attrs[attr_type]:
                names.append((a["idx"], a["name"]))
    names.sort(key=lambda x: x[0])
    return [n for _, n in names]

feature_names = get_feature_names(full_predictions)
print(f"Feature count: {len(feature_names)}")

# ── Spark feature importances — from the FINAL (full-population) model ──
importances = rf_final_model.featureImportances.toArray()
importance_df = spark.createDataFrame(
    [(feature_names[i], float(importances[i])) for i in range(len(feature_names))],
    ["feature", "importance"]
).orderBy(F.desc("importance"))

display(importance_df)

# ── SHAP values via shadow sklearn RF — trained on FULL population ──
features_pd = full_predictions.select(
    "bill_id",
    "target",
    vector_to_array(col("features")).alias("features_arr")
).toPandas()

X = pd.DataFrame(features_pd["features_arr"].tolist(), columns=feature_names)
y = features_pd["target"]

shadow_rf = SklearnRF(
    n_estimators=rf_final_model.getNumTrees,
    max_depth=rf_final_model.getOrDefault("maxDepth"),
    random_state=42
)
shadow_rf.fit(X, y)

explainer = shap.TreeExplainer(shadow_rf)
shap_values = explainer.shap_values(X)
pass_shap = shap_values[1] if isinstance(shap_values, list) else shap_values

shap_long = pd.DataFrame(pass_shap, columns=feature_names)
shap_long.insert(0, "bill_id", features_pd["bill_id"])
shap_long = shap_long.melt(id_vars="bill_id", var_name="feature", value_name="shap_value")

top_reasons = (
    shap_long.assign(abs_val=shap_long["shap_value"].abs())
    .sort_values(["bill_id", "abs_val"], ascending=[True, False])
    .groupby("bill_id")
    .head(5)
    .drop(columns="abs_val")
)

# ── Write everything to Fabric gold layer — all FULL-population now ──
spark.sql("CREATE SCHEMA IF NOT EXISTS congress_gold")

spark.sql("DROP TABLE IF EXISTS congress_gold.bill_predictions")
full_predictions.withColumn("prob_pass", vector_to_array(col("probability"))[1]) \
    .select("bill_id", "target", "prediction", "prob_pass") \
    .write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("congress_gold.bill_predictions")

importance_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable("congress_gold.bill_feature_importance")

spark.createDataFrame(shap_long).write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable("congress_gold.bill_shap_values")

spark.createDataFrame(top_reasons).write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true").saveAsTable("congress_gold.bill_shap_top_reasons")

print(f"  Validation (test-only) AUC: {auc:.3f} | Accuracy: {accuracy:.3f}")
print("  Registered: congress_gold.bill_predictions (full population)")
print("  Registered: congress_gold.bill_feature_importance (full-population model)")
print("  Registered: congress_gold.bill_shap_values (full population)")
print("  Registered: congress_gold.bill_shap_top_reasons (full population)")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
