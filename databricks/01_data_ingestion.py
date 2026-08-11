from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("AvionicsMaintenance").getOrCreate()

# Define column names for CMAPSS dataset
columns = ["engine_id", "cycle", "setting1", "setting2", "setting3"] + \
          [f"sensor_{i}" for i in range(1, 22)]

# Read raw file — Spark initially detects 28 columns due to trailing 
# spaces in the source file creating 2 extra empty columns
df_raw = spark.read.csv(
    "/Volumes/workspace/default/sensor_data/train_FD001.txt",
    sep=" ",
    inferSchema=True
)

print(f"Actual number of columns detected: {len(df_raw.columns)}")
df_raw.show(3)

# Drop the 2 trailing empty columns (_c26, _c27) caused by trailing 
# spaces in the source file, then apply proper column names
df_train = df_raw.select(df_raw.columns[:26]).toDF(*columns)

df_train.describe().show()
print(f"Total records: {df_train.count()}")
print(f"Unique engines: {df_train.select('engine_id').distinct().count()}")

df_train.write.mode("overwrite").saveAsTable("avionics_raw")