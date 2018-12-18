package pagerank;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Cluster;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Job;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.io.IOException;


public class ParseReducer extends Reducer<Text, Text, Text, Text> { 
	
	private HashSet<String> all_title;

	protected void setup(Context context) throws IOException, InterruptedException {
		all_title = new HashSet<String>();	
	
		Configuration conf = context.getConfiguration();
		Cluster cluster = new Cluster(conf);
		Job job = cluster.getJob(context.getJobID());
		long N = job.getCounters().findCounter(PageRank.PAGE_RANK_COUNTER.N).getValue();
		long D = job.getCounters().findCounter(PageRank.PAGE_RANK_COUNTER.Dangling).getValue();	
	}

	public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {				
	
		StringBuffer links_buf = new StringBuffer();
		double rank = (all_title.size()==0)? 0 : ((double)1) / all_title.size();
	
		links_buf.append(String.valueOf(rank));
		links_buf.append("|");
		
		String key_str = key.toString();
		for (Text val: values) {
			String val_str = val.toString();
			if(key_str.charAt(0) == ' ') {			
				all_title.add(val_str);
			}
			else {
				if(all_title.contains(val_str)) {				
					links_buf.append(val_str);
					links_buf.append("|");
				}
			}
		}		

		Text value = new Text(links_buf.toString());
		if (rank != 0) {
			context.write(key, value);		
		}
	
	}
}
