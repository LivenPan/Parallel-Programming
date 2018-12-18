package pagerank;
import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Cluster;
import org.apache.hadoop.mapreduce.Job;

import java.util.ArrayList;
import java.util.Arrays;
import java.net.URI; 
import java.io.*;

public class ParseMapper extends Mapper<LongWritable, Text, Text, Text> {
	
	private int n_reducer;
	private static long N, Dangling;

	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
			Configuration conf = context.getConfiguration();
			n_reducer = conf.getInt("n_reducer", 0);			
			String input = unescapeXML(value.toString()); 		
			/*  Match title pattern */  
			Pattern titlePattern = Pattern.compile("<title>(.+?)</title>");
			Matcher titleMatcher = titlePattern.matcher(input);
			// No need capitalizeFirstLetter
			titleMatcher.find();
			String title = titleMatcher.group(1);
			for (int i = 0; i < n_reducer; i++) {
				Text k = new Text();
				Text v = new Text();
				k.set(" " + Integer.toString(i));
				v.set(title);
				context.write(k, v);
			}

			/*  Match link pattern */
			Pattern linkPattern = Pattern.compile("\\[\\[(.+?)([\\|#]|\\]\\])");
			Matcher linkMatcher = linkPattern.matcher(input);
			// Need capitalizeFirstLetter

			if(linkMatcher.find()) {
				String link = capitalizeFirstLetter(linkMatcher.group(1));
				Text k = new Text(title);
				Text v = new Text(link);
				context.write(k, v);
			}
			else { 				
		
				Text k = new Text(title);
				Text v = new Text("|}*{|");
				context.write(k, v);
			}

			while(linkMatcher.find()) {
                String link = capitalizeFirstLetter(linkMatcher.group(1));
				Text k = new Text(title);
                Text v = new Text(link);
                context.write(k, v);
			}
			N++;			
	}
	
	private String unescapeXML(String input) {

		return input.replaceAll("&lt;", "<").replaceAll("&gt;", ">").replaceAll("&amp;", "&").replaceAll("&quot;", "\"").replaceAll("&apos;", "\'");

    }

    private String capitalizeFirstLetter(String input){

    	char firstChar = input.charAt(0);

        if ( firstChar >= 'a' && firstChar <='z'){
            if ( input.length() == 1 ){
                return input.toUpperCase();
            }
            else
                return input.substring(0, 1).toUpperCase() + input.substring(1);
        }
        else 
        	return input;
    }

	protected void cleanup(Context context) throws IOException, InterruptedException {	
 		context.getCounter(PageRank.PAGE_RANK_COUNTER.N).increment(N); 			
	}
}