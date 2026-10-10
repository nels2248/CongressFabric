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
# META       "known_warehouses": []
# META     }
# META   }
# META }

# MARKDOWN ********************

# Config

# CELL ********************

LAKEHOUSE_PATH = "abfss://3961b5ec-d383-4029-b4d8-9c4b4acd3d4c@onelake.dfs.fabric.microsoft.com/39418109-5bb5-4630-a82f-fc5836ac8a08/Files/congress"
BETWEENNESS_SAMPLE_K = 200  # sampled nodes for betweenness centrality (speed vs. accuracy tradeoff)

print("Cell 1 loaded — config ready")
print(f"  LAKEHOUSE_PATH = {LAKEHOUSE_PATH}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************


# MARKDOWN ********************

# Imports

# CELL ********************

import networkx as nx
import pandas as pd
from pyspark.sql import functions as F
from networkx.algorithms.community import greedy_modularity_communities

print("Cell 2 loaded — libraries imported")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************


# MARKDOWN ********************

# Load sponsor and co sponsor

# CELL ********************

print("Loading sponsor and cosponsor data from Silver layer...")

cosponsors_df = spark.table("congress_silver.cosponsors").select(
    "bill_id", "bioguide_id", "full_name", "party"
).toPandas()

sponsors_df = spark.table("congress_silver.sponsors").select(
    "bill_id", "bioguide_id", "full_name", "party"
).toPandas()

print(f"  sponsors:   {len(sponsors_df):,} rows")
print(f"  cosponsors: {len(cosponsors_df):,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Build weighted edge list

# MARKDOWN ********************


# CELL ********************


edges = sponsors_df.merge(cosponsors_df, on="bill_id", suffixes=("_sponsor", "_cosponsor"))
edges = edges[["bioguide_id_sponsor", "bioguide_id_cosponsor"]].dropna()
edges.columns = ["source", "target"]

edge_weights = edges.groupby(["source", "target"]).size().reset_index(name="weight")

print(f"✓ Edge list built — {len(edge_weights):,} unique collaboration pairs")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# Build graph, centrality, communities

# CELL ********************

print("Building graph...")
G = nx.from_pandas_edgelist(edge_weights, "source", "target", edge_attr="weight")
print(f"  {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

print("Computing centrality measures...")
degree_cent = nx.degree_centrality(G)
betweenness_cent = nx.betweenness_centrality(G, weight="weight", k=BETWEENNESS_SAMPLE_K)
eigen_cent = nx.eigenvector_centrality(G, weight="weight", max_iter=500)

print("Detecting communities (greedy modularity)...")
communities = greedy_modularity_communities(G, weight="weight")
community_map = {member: i for i, group in enumerate(communities) for member in group}
print(f"  {len(communities)} communities detected")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************


# MARKDOWN ********************

#  Assemble node table

# CELL ********************

member_lookup = pd.concat([sponsors_df, cosponsors_df]).drop_duplicates("bioguide_id").set_index("bioguide_id")

nodes_out = pd.DataFrame({"bioguide_id": list(G.nodes())})
nodes_out["full_name"] = nodes_out["bioguide_id"].map(member_lookup["full_name"])
nodes_out["party"] = nodes_out["bioguide_id"].map(member_lookup["party"])
nodes_out["degree_centrality"] = nodes_out["bioguide_id"].map(degree_cent)
nodes_out["betweenness_centrality"] = nodes_out["bioguide_id"].map(betweenness_cent)
nodes_out["eigenvector_centrality"] = nodes_out["bioguide_id"].map(eigen_cent)
nodes_out["community_id"] = nodes_out["bioguide_id"].map(community_map)

print(f"✓ Node table assembled — {len(nodes_out):,} legislators")
display(nodes_out.sort_values("betweenness_centrality", ascending=False).head(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Cell 6b — Enrich edges with source/target party + name for Power BI (avoids relationship gymnastics)
node_attrs = nodes_out.set_index("bioguide_id")[["full_name", "party"]]

edge_weights = edge_weights.merge(
    node_attrs.rename(columns={"full_name": "source_name", "party": "source_party"}),
    left_on="source", right_index=True, how="left"
)

edge_weights = edge_weights.merge(
    node_attrs.rename(columns={"full_name": "target_name", "party": "target_party"}),
    left_on="target", right_index=True, how="left"
)

print(f"✓ Edges enriched — {len(edge_weights):,} rows now include source/target name + party")
display(edge_weights.head())

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************


# MARKDOWN ********************

# Cleanp data first

# MARKDOWN ********************

# Wrrite to gold and register

# CELL ********************

print("Writing Gold tables...")

spark.createDataFrame(nodes_out).write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("congress_gold.network_nodes")

spark.createDataFrame(edge_weights).write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("congress_gold.network_edges")

print("  Registered: congress_gold.network_nodes")
print("  Registered: congress_gold.network_edges")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************


# MARKDOWN ********************


# MARKDOWN ********************

# Validation Summary

# CELL ********************

print("=" * 55)
print("NETWORK ANALYSIS VALIDATION SUMMARY")
print("=" * 55)

spark.sql("SELECT party, COUNT(*) AS members FROM congress_gold.network_nodes GROUP BY party ORDER BY members DESC").show()
spark.sql("SELECT community_id, COUNT(*) AS members FROM congress_gold.network_nodes GROUP BY community_id ORDER BY members DESC LIMIT 10").show()
spark.sql("SELECT COUNT(*) AS total_edges, SUM(weight) AS total_collaborations FROM congress_gold.network_edges").show()

print("Network analysis complete. Tables available in:")
print("  • SQL Analytics Endpoint: congress_gold.network_nodes, congress_gold.network_edges")
print("  • Power BI DirectLake: point at congress_gold schema")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

