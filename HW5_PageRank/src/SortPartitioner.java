package pagerank;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.io.NullWritable;
import java.io.IOException;

public class SortPartitioner extends Partitioner<Text, Text> {

	private double Max, Min;
	private int N = 1;
	private double avg = 0;

    @Override
    public int getPartition(Text key, Text value, int numReduceTasks) {
		if(key.toString().equals(" ")) {
			N = Integer.parseInt(value.toString());
			avg = ((double)1) / N;			
			return 0;
		}
		else {
			double rank = Double.parseDouble(value.toString());
			if(rank > avg) return 0;
			else  return 1;
		}
    }
}
