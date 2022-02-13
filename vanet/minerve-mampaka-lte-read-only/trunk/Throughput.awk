##################################################################################
# Throughput.awk script of 						         #
# Development of a simulation and performance analysis platform for LTE networks #
# Project done by MINERVE MAMPAKA 					         #
# December 2013								         #
##################################################################################


BEGIN {
	
	#create filenames according to input parameters
	FilenameRTP="ThroughputRTP.txt";
	FilenameCBR="ThroughputCBR.txt";
	FilenameHTTP="ThroughputHTTP.txt";
	FilenameFTP="ThroughputFTP.txt";	
	FilenameTotal="ThroughputTotal.txt";
	
	#initialize variables	
	for (i=0; i<=3600; i++) {
		HTTP[i]=0;
		CBR[i]=0;
		FTP[i]=0;
		RTP[i]=0;
		Total[i]=0;
	}
}
{

#save traces file results in different variables
#r  0.241408 1  0  tcp  1040 -------  1     4.0   0.0   3    6
#$1 $2       $3 $4 $5   $6      $7    $8    $9    $10   $11  $12

   action 	= $1;
   time 	= $2;
   from 	= $3;
   to 		= $4;
   type		= $5;
   pktsize 	= $6;
   flow_id 	= $8;
   src 		= $9;
   dst 		= $10;
   seq_no 	= $11;
   packet_id 	= $12;


#calculate the throughput of the EnodeB with node ID 0
#whenever the event is a packet reception and the receiving node is the EnodeB
#every second sum of packet size giving throughput for different types and  total throughput are saved

        if (from==0 && action=="r") {
		TimeIndex=sprintf("%d",time);
		if (flow_id == 0) {			
			RTP[TimeIndex] = RTP[TimeIndex] + pktsize;
		}
		if (flow_id == 1) {
			CBR[TimeIndex] = CBR[TimeIndex] + pktsize;
		}
		if (flow_id == 2) {
			HTTP[TimeIndex] = HTTP[TimeIndex] + pktsize;
		}
		if (flow_id == 3) {
			FTP[TimeIndex] = FTP[TimeIndex] + pktsize;
		}
		Total[TimeIndex] = Total[TimeIndex] + pktsize;
	}
}	

#write the throughputs in the respective files based on different traffic types
#the throughputs generated are expressed in Mbit/s

END {
 	for (i=0; i<time; i++) {
		printf("%d %f\n", i+1, RTP[i]*8/(1024*1024)) >> "Throughput/" FilenameRTP
		printf("%d %f\n", i+1, CBR[i]*8/(1024*1024)) >> "Throughput/" FilenameCBR
		printf("%d %f\n", i+1, HTTP[i]*8/(1024*1024)) >> "Throughput/" FilenameHTTP
		printf("%d %f\n", i+1, FTP[i]*8/(1024*1024)) >> "Throughput/" FilenameFTP
		printf("%d %f\n", i+1, Total[i]*8/(1024*1024)) >> "Throughput/" FilenameTotal
	}	
}

