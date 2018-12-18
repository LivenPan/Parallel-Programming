package pagerank;
import java.io.IOException;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.NullWritable;

public class SortReducer extends Reducer<Text, Text, Text, Text> {

    public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
     

		if(key.toString().equals(" ")) return;        

        for (Text val: values) {
			String key_str = key.toString();
			String k_str = key_str.split("\\|", 2)[0];
			Text k = new Text(k_str);
            context.write(k, val);
        }       
    }
}
