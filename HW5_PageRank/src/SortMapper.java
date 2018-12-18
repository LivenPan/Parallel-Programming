package pagerank;
import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Cluster;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Job;

public class SortMapper extends Mapper<Text, Text, Text, Text> {

	private static double max = -1000, min = 1000;
	private double sum = 0;
	
	protected void setup(Context context) throws IOException, InterruptedException {
		Configuration conf = context.getConfiguration();
        int N = conf.getInt("N", 0);	
		Text k = new Text(" ");
		Text v = new Text(String.valueOf(N));
		context.write(k, v);
	}

    public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
		String value_str = value.toString();
		String[] value_arr = value_str.split("\\|");
		
		Double rank = Double.parseDouble(value_arr[0]);	
	
		Text k = new Text(key.toString()+"|"+value_arr[0]);
		Text v = new Text(value_arr[0]);
		context.write(k, v);
    }	


}
