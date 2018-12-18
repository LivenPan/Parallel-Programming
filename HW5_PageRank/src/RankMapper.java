package pagerank;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Cluster;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Job;

import java.io.IOException;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.ArrayList;
import java.util.Arrays;
import java.net.URI; 
import java.io.*;

public class RankMapper extends Mapper<Text, Text, Text, Text> {	
	
	private static double DanglingSum;
	private long N, Dangling;

	protected void setup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
		N = conf.getInt("N", 0);	
        
    }


	public void map(Text key, Text value, Context context) throws IOException, InterruptedException {
		Configuration conf = context.getConfiguration();
		
		String value_str = value.toString();
		String[] value_arr = value_str.split("\\|");
		
		if(value_arr.length == 1) {	
			Double rank = Double.parseDouble(value_arr[0]);
			DanglingSum += rank;
			Dangling++;
		}
		else if(value_arr.length > 1) {
			Double rank = Double.parseDouble(value_arr[0]);				
			Double new_rank = rank / (value_arr.length-1);
			Text v = new Text("!|" + String.valueOf(new_rank));

			for(int i=1; i<value_arr.length; i++) {
				Text k = new Text(value_arr[i]);					
				context.write(k, v);
			}

			if(key.toString().equals("Autonomica")) {
				for(int i=1; i<value_arr.length; i++) {
					System.out.printf("Error");
				}
			}
				
		}	
		else {
			System.out.println("Error!!!");
		}
		context.write(key, value);
	}

	protected void cleanup(Context context) throws IOException, InterruptedException {
		Configuration conf = context.getConfiguration();
	
		context.getCounter(PageRank.PAGE_RANK_COUNTER.Dangling).increment(Dangling);
		context.getCounter(PageRank.PAGE_RANK_COUNTER.DanglingSum).increment((long)(DanglingSum*1E18));
	}

}
