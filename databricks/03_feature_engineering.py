from pyspark.sql.functions import max as spark_max, col, avg, stddev, when
from pyspark.sql.window import Window

# Reload the table
df_train = spark.table("avionics_raw")

# Calculate Remaining Useful Life (RUL) - the key target variable
max_cycles = df_train.groupBy("engine_id").agg(spark_max("cycle").alias("max_cycle"))

df_with_rul = df_train.join(max_cycles, on="engine_id") \
                      .withColumn("RUL", col("max_cycle") - col("cycle")) \
                      .drop("max_cycle")

# Create rolling window features (moving averages over last 5 cycles)
window_spec = Window.partitionBy("engine_id").orderBy("cycle").rowsBetween(-4, 0)

for sensor in [f"sensor_{i}" for i in range(1, 22)]:
    df_with_rul = df_with_rul.withColumn(
        f"{sensor}_rolling_avg", avg(col(sensor)).over(window_spec)
    )

# Create failure flag (engine within 30 cycles of failure)
df_with_rul = df_with_rul.withColumn(
    "failure_imminent", when(col("RUL") <= 30, 1).otherwise(0)
)

print(f"Feature engineered dataset: {df_with_rul.count()} rows, {len(df_with_rul.columns)} columns")
df_with_rul.select("engine_id", "cycle", "RUL", "failure_imminent").show(10)

df_with_rul.write.mode("overwrite").saveAsTable("avionics_features")

# Select a manageable subset of columns for the Snowflake demo
sample_df = df_with_rul.select(
    "engine_id", "cycle", "setting1", "setting2",
    "sensor_2", "sensor_3", "sensor_4", "sensor_7",
    "RUL", "failure_imminent"
).limit(2000)

sample_pandas = sample_df.toPandas()
sample_pandas.to_csv("/Volumes/workspace/default/sensor_data/avionics_sample.csv", index=False)
print(f"Sample exported: {sample_pandas.shape[0]} rows, {sample_pandas.shape[1]} columns")