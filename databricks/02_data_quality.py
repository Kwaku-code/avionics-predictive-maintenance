from pyspark.sql.functions import isnan, when, count, col

# Reload the table saved in the previous notebook
df_train = spark.table("avionics_raw")

missing_counts = df_train.select([
    count(when(isnan(c) | col(c).isNull(), c)).alias(c)
    for c in df_train.columns
])
missing_counts.show()

duplicates = df_train.count() - df_train.dropDuplicates().count()
print(f"Duplicate records: {duplicates}")