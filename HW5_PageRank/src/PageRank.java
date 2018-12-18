package pagerank;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.lib.input.KeyValueTextInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.Job;
import java.io.*;

public class PageRank {

  public static int Iteration = 0;
	private static final int N_Reducer = 32;
   
	public static enum PAGE_RANK_COUNTER {
		N, Dangling, DanglingSum, Error, Max, Min, Sum
	}

  public static void main(String[] args) throws Exception {
	
  // If user type the number for iteration
  if(args.length > 4) 
	  Iteration = Integer.parseInt(args[4]); 
 
  // Parsing
	Configuration ConfParsing = new Configuration();
  ConfParsing.setInt("n_reducer", N_Reducer);

  Job JobForParsing = Job.getInstance(ConfParsing, "Parse");
  JobForParsing.setJarByClass(PageRank.class);

  // set the class of each stage in mapreduce
  JobForParsing.setMapperClass(ParseMapper.class);
  JobForParsing.setPartitionerClass(ParsePartitioner.class);
  JobForParsing.setReducerClass(ParseReducer.class);

  // set the output class of Mapper and Reducer
  JobForParsing.setMapOutputKeyClass(Text.class);
  JobForParsing.setMapOutputValueClass(Text.class);
  JobForParsing.setOutputKeyClass(Text.class);
  JobForParsing.setOutputValueClass(Text.class);

  // set the number of reducer
  JobForParsing.setNumReduceTasks(N_Reducer);

  // add input/output path
  FileInputFormat.addInputPath(JobForParsing, new Path(args[0]));
  FileOutputFormat.setOutputPath(JobForParsing, new Path(args[1]));

  JobForParsing.waitForCompletion(true);		
		
	int N = (int)JobForParsing.getCounters().findCounter(PageRank.PAGE_RANK_COUNTER.N).getValue();
		
		
	// Ranking
	
	if(args.length > 4)	{
		for(int i = 0; i < Iteration; i++) {
	  	Configuration ConfRanking = new Configuration();
			ConfRanking.setInt("N", N);

			Job forRanking = Job.getInstance(ConfRanking, "Rank");
			forRanking.setJarByClass(PageRank.class);
			forRanking.setNumReduceTasks(N_Reducer);		
			forRanking.setInputFormatClass(KeyValueTextInputFormat.class);

			// Input for mapper~
			forRanking.setMapperClass(RankMapper.class);			
			forRanking.setReducerClass(RankReducer.class);

			// Output for reducer~
			forRanking.setMapOutputKeyClass(Text.class);
			forRanking.setMapOutputValueClass(Text.class);
			forRanking.setOutputKeyClass(Text.class);
			forRanking.setOutputValueClass(Text.class);

      // Input,Output path setting
			FileInputFormat.addInputPath(forRanking, new Path(args[1]));
			FileOutputFormat.setOutputPath(forRanking, new Path(args[2]));
			forRanking.waitForCompletion(true);		
				
			//Calculating	
			double Error = ((double)(forRanking.getCounters().findCounter(PageRank.PAGE_RANK_COUNTER.Error).getValue()))/1E18; 
				
			if(Error < 0.001 || (i+1 == Iteration)) 
				break;
			else {
				FileSystem FS = FileSystem.get(ConfRanking);
				FS.delete(new Path(args[1]), true);
				FS.rename(new Path(args[2]), new Path(args[1]));
			}				
		}
	}
		
	else{		
	  for(int i = 0; ; i++) {
		  Configuration ConfRanking = new Configuration();
			ConfRanking.setInt("N", N);

			Job forRanking = Job.getInstance(ConfRanking, "Rank");
			forRanking.setJarByClass(PageRank.class);
			forRanking.setNumReduceTasks(N_Reducer);		
			forRanking.setInputFormatClass(KeyValueTextInputFormat.class);

			// Input for mapper~
			forRanking.setMapperClass(RankMapper.class);			
			forRanking.setReducerClass(RankReducer.class);

			// Output for reducer~
			forRanking.setMapOutputKeyClass(Text.class);
			forRanking.setMapOutputValueClass(Text.class);
			forRanking.setOutputKeyClass(Text.class);
			forRanking.setOutputValueClass(Text.class);

	    // Input,Output path setting
			FileInputFormat.addInputPath(forRanking, new Path(args[1]));
			FileOutputFormat.setOutputPath(forRanking, new Path(args[2]));
			forRanking.waitForCompletion(true);		
				
			//Calculating		
			double Error = ((double)(forRanking.getCounters().findCounter(PageRank.PAGE_RANK_COUNTER.Error).getValue()))/1E18; 
			
			if(Error < 0.001) 
			  break;
			else {
				FileSystem FS = FileSystem.get(ConfRanking);
				FS.delete(new Path(args[1]), true);
				FS.rename(new Path(args[2]), new Path(args[1]));
			}			
		}			
	}
		
		
	//Sorting
	Configuration ConfSorting = new Configuration();
				
	Job forSorting = Job.getInstance(ConfSorting, "Sort");
  ConfSorting.setInt("n_reducer", N_Reducer);
	forSorting.setJarByClass(PageRank.class);
	forSorting.setInputFormatClass(KeyValueTextInputFormat.class);

	// Input for mapper~
	forSorting.setMapperClass(SortMapper.class);		
	forSorting.setSortComparatorClass(SortKeyComparator.class);
	forSorting.setReducerClass(SortReducer.class);
		
	// Output for reducer~
	forSorting.setMapOutputKeyClass(Text.class);
  forSorting.setMapOutputValueClass(Text.class);
  forSorting.setOutputKeyClass(Text.class);
  forSorting.setOutputValueClass(Text.class);

	// Input,Output path setting
	FileInputFormat.addInputPath(forSorting, new Path(args[2]));
  FileOutputFormat.setOutputPath(forSorting, new Path(args[3]));
  forSorting.waitForCompletion(true);
	
	System.exit(0);
  }
}