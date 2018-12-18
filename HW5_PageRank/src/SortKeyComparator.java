package pagerank;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.WritableComparator;

public class SortKeyComparator extends WritableComparator {

	public SortKeyComparator() {
		super(Text.class, true);
	}

	public int compare(WritableComparable o1, WritableComparable o2) {
		Text key1 = (Text) o1;
		Text key2 = (Text) o2;
		
		String key1_str = key1.toString();
		String key2_str = key2.toString();		
	
		if(key1_str.equals(" ")) return -1;
		else if(key2_str.equals(" ")) return 1;
		
		String[] key1_arr = key1_str.split("\\|");
		String[] key2_arr = key2_str.split("\\|");

		String title1 = key1_arr[0];
		Double rank1 = Double.parseDouble(key1_arr[1]);
		String title2 = key2_arr[0];
		Double rank2 = Double.parseDouble(key2_arr[1]);

		if(rank1 > rank2) return -1;
		else if (rank1 < rank2) return 1;
		else return title1.compareTo(title2);
	}
}
		

		
