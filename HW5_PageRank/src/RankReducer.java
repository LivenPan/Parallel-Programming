package pagerank;

import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Cluster;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.io.Text;

import java.io.IOException;
import java.lang.Math;

public class RankReducer extends Reducer<Text, Text, Text, Text> {

	private Job rr_job;	
	private long N, Dangling;
	private double DanglingSum, Error;
	private static final double damping = 0.85;

	protected void setup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        Cluster cluster = new Cluster(conf);
        rr_job = cluster.getJob(context.getJobID());
		N = conf.getInt("N", 0);
		Dangling = rr_job.getCounters().findCounter(PageRank.PAGE_RANK_COUNTER.Dangling).getValue();
		DanglingSum = ((double)(rr_job.getCounters().findCounter(PageRank.PAGE_RANK_COUNTER.DanglingSum).getValue()))/1E18;	
    }	

	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
		double rank = 0, rank_last = 0, rank_final = 0;
		String links_last = "";

		for (Text val: values) {
			String value_str = val.toString();
			String[] value_arr = value_str.split("\\|");
			
			if(value_arr[0].equals("!")) {	
				if(value_arr.length == 1)  System.out.println("Error!!!");
				else {
					Double rank_tmp = Double.parseDouble(value_arr[1]);
					rank += rank_tmp;								
				}
			}
			else {	
				rank_last = Double.parseDouble(value_arr[0]);
				links_last = value_str.split("\\|", 2)[1];
			}
			
		}

		rank_final = ((double)(1-damping))/N + damping*rank + damping*DanglingSum/N;  			
		double local_error = Math.abs(rank_final - rank_last);		
	
		Error += local_error;	

		String rank_str = String.valueOf(rank_final) + "|";
		Text v = new Text(rank_str+links_last);
		context.write(key, v);
	}

	protected void cleanup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        context.getCounter(PageRank.PAGE_RANK_COUNTER.Error).increment((long)(Error*1E18));
    }
}

