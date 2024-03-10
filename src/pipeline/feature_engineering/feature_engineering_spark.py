from pyspark.sql.window import Window
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date, col, rank, avg, floor, current_date, datediff, udf
import time
import argparse

class FeatureSet:
    def __init__(self) -> None:
        self.spark = SparkSession.builder.appName("pyspark_window").getOrCreate()
        self.window_partition = Window.partitionBy("id").orderBy("loan_date")
        self.df = None
    
    def load_data(self, file, header=True) -> None:
        self.df = self.spark.read.option("header", header).csv(file)

    def process_new_features(self) -> None:
        self.df = self.df.withColumn("loan_date", to_date(col("loan_date"), "yyyy-MM-dd"))

        self.df = self.df.withColumn("nb_previous_loans", rank().over(self.window_partition) - 1)

        self.df = self.df.withColumn("avg_amount_loans_previous", avg("loan_amount").over(self.window_partition.rowsBetween(Window.unboundedPreceding,-1)))

        self.df = self.df.withColumn("birthday", to_date(col("birthday"), "yyyy-MM-dd"))
        self.df = self.df.withColumn("job_start_date", to_date(col("job_start_date"), "yyyy-MM-dd"))

        self.df = self.df.withColumn("age", floor(datediff(current_date(), to_date(col("birthday"), "yyyy-MM-dd"))/365))
        self.df = self.df.withColumn("years_on_the_job", floor(datediff(current_date(), to_date(col("job_start_date"), "yyyy-MM-dd"))/365))

        udf_flag_own_car = udf(lambda x: 0 if x == 'N' else 1) 
        self.df = self.df.withColumn("flag_own_car", udf_flag_own_car(col("flag_own_car")))

    def save_data(self, columns, folder) -> None:
        self.df = self.df.select(columns)

        self.df.select(columns).write.parquet(folder)

def parse_args():
    parser = argparse.ArgumentParser(description="Script inputs and outputs")
    parser.add_argument("--input_data", type=str, help="Input data location")
    parser.add_argument("--output_data", type=str, help="Output data location")
    return parser.parse_args()

def main():
    args = parse_args()

    feature_set = FeatureSet()

    feature_set.load_data(args.input_data)
    feature_set.process_new_features()

    columns = ['id', 'age', 'years_on_the_job', 'nb_previous_loans', 'avg_amount_loans_previous', 'flag_own_car', 'status']
    feature_set.save_data(columns, args.output_data)

if __name__ == '__main__':

    start_time = time.time()
    main()
    end_time = time.time()

    print("Total Time:", end_time - start_time, "seconds.")
