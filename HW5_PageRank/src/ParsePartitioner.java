package pagerank;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Partitioner;

import java.io.*;

public class ParsePartitioner extends Partitioner<Text, Text> {
    @Override
    public int getPartition(Text key, Text value, int numReduceTasks) {
       
        char start = key.toString().charAt(0);
        if(start == ' ') {
			String index_str = key.toString().substring(1); 
			int index = Integer.parseInt(index_str);
			if(index < 0 || index > 31) System.out.println("Error!!!!");	
            return index;
		}
        else {
		
			int id = (key.hashCode() & Integer.MAX_VALUE) % numReduceTasks;
			return id;
    	}
	}
}
